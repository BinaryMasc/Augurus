using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

namespace Augurus.Api;

public class InferenceEngine
{
    private readonly InferenceSession _session;
    private readonly List<double> _binEdges;

    public InferenceEngine(string modelPath, string quantizerPath)
    {
        _session = new InferenceSession(modelPath);
        
        var json = File.ReadAllText(quantizerPath);
        var doc = JsonDocument.Parse(json);
        var edgesArray = doc.RootElement.GetProperty("bin_edges").EnumerateArray();
        
        _binEdges = new List<double>();
        foreach (var edge in edgesArray)
        {
            _binEdges.Add(edge.GetDouble());
        }
    }

    public PredictionResponse Predict(List<Candle> Candles)
    {
        if (Candles.Count != ModelConfig.SeqLen)
            throw new ArgumentException($"Need exactly {ModelConfig.SeqLen} candles, got {Candles.Count}.");

        // 1. Calculate Log Returns & Simplified Rolling Volatility
        var logReturns = new double[Candles.Count];
        for (int i = 1; i < Candles.Count; i++)
        {
            logReturns[i] = Math.Log(Candles[i].Close / Candles[i - 1].Close);
        }

        double mean = logReturns.Skip(1).Average();
        double sumSq = logReturns.Skip(1).Sum(r => Math.Pow(r - mean, 2));
        double rollingVol = Math.Sqrt(sumSq / (logReturns.Length - 1));
        if (rollingVol == 0) rollingVol = 1e-8;

        // 2. Prepare Tensors
        var tokens = new long[1, ModelConfig.SeqLen];
        var continuous = new float[1, ModelConfig.SeqLen, ModelConfig.NumContinuousFeatures];

        for (int i = 0; i < ModelConfig.SeqLen; i++)
        {
            var candle = Candles[i];
            
            // Normalize return
            double r = Math.Log(candle.Close / (i == 0 ? candle.Open : Candles[i-1].Close));
            double normReturn = r / rollingVol;

            tokens[0, i] = GetToken(normReturn);

            // Time embeddings
            continuous[0, i, 0] = (float)Math.Sin(2 * Math.PI * candle.Timestamp.Hour / 24.0);
            continuous[0, i, 1] = (float)Math.Cos(2 * Math.PI * candle.Timestamp.Hour / 24.0);
            continuous[0, i, 2] = (float)Math.Sin(2 * Math.PI * (int)candle.Timestamp.DayOfWeek / 7.0);
            continuous[0, i, 3] = (float)Math.Cos(2 * Math.PI * (int)candle.Timestamp.DayOfWeek / 7.0);
            continuous[0, i, 4] = (float)normReturn;
        }

        List<double> predictedPrices = new();
        double currentClose = Candles.Last().Close;
        DateTime currentTime = Candles.Last().Timestamp;

        // 3. Autoregressive Loop for 5 future candles
        for (int step = 0; step < 5; step++)
        {
            var tokensTensor = new DenseTensor<long>(tokens.Cast<long>().ToArray(), new[] { 1, ModelConfig.SeqLen });
            var contTensor = new DenseTensor<float>(continuous.Cast<float>().ToArray(), new[] { 1, ModelConfig.SeqLen, ModelConfig.NumContinuousFeatures });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("tokens", tokensTensor),
                NamedOnnxValue.CreateFromTensor("continuous_features", contTensor)
            };

            using var results = _session.Run(inputs);
            
            // Use Phase 1 output (Next-Token Prediction)
            var nextTokenLogits = results.First(r => r.Name == "next_token_logits").AsTensor<float>();

            // Find argmax token for the very last step in the sequence
            int bestToken = 0;
            float maxLogit = float.MinValue;
            for (int i = 0; i < ModelConfig.VocabSize; i++)
            {
                float logit = nextTokenLogits[0, ModelConfig.SeqLen - 1, i];
                if (logit > maxLogit)
                {
                    maxLogit = logit;
                    bestToken = i;
                }
            }

            // Un-quantize token to $ Return
            double predictedNormReturn = GetNormReturnFromToken(bestToken);
            double predictedLogReturn = predictedNormReturn * rollingVol;
            double nextClose = currentClose * Math.Exp(predictedLogReturn);
            predictedPrices.Add(nextClose);

            // Shift inputs left to make room for the new prediction
            for (int i = 0; i < ModelConfig.SeqLen - 1; i++)
            {
                tokens[0, i] = tokens[0, i + 1];
                for (int j = 0; j < ModelConfig.NumContinuousFeatures; j++)
                    continuous[0, i, j] = continuous[0, i + 1, j];
            }

            // Append new state at sequence end
            currentTime = currentTime.AddMinutes(1);
            tokens[0, ModelConfig.SeqLen - 1] = bestToken;
            continuous[0, ModelConfig.SeqLen - 1, 0] = (float)Math.Sin(2 * Math.PI * currentTime.Hour / 24.0);
            continuous[0, ModelConfig.SeqLen - 1, 1] = (float)Math.Cos(2 * Math.PI * currentTime.Hour / 24.0);
            continuous[0, ModelConfig.SeqLen - 1, 2] = (float)Math.Sin(2 * Math.PI * (int)currentTime.DayOfWeek / 7.0);
            continuous[0, ModelConfig.SeqLen - 1, 3] = (float)Math.Cos(2 * Math.PI * (int)currentTime.DayOfWeek / 7.0);
            continuous[0, ModelConfig.SeqLen - 1, 4] = (float)predictedNormReturn;

            currentClose = nextClose;
        }

        return new PredictionResponse
        {
            PredictedClosePrices = predictedPrices
        };
    }

    private long GetToken(double normReturn)
    {
        for (int b = 0; b < _binEdges.Count - 1; b++)
        {
            if (normReturn >= _binEdges[b] && normReturn < _binEdges[b + 1])
                return b;
        }
        return ModelConfig.VocabSize - 1;
    }

    private double GetNormReturnFromToken(int token)
    {
        // Handle edges carefully
        if (token == 0) return _binEdges[1] - 0.1;
        if (token == ModelConfig.VocabSize - 1) return _binEdges[ModelConfig.VocabSize - 1] + 0.1;
        
        // Return the center value of the probability bin
        return (_binEdges[token] + _binEdges[token + 1]) / 2.0;
    }
}
