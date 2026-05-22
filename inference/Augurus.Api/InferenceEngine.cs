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

        // Calculate EMAs recursively (matches Python's ewm(adjust=False))
        double[] ema12 = new double[Candles.Count];
        double[] ema26 = new double[Candles.Count];
        double alpha12 = 2.0 / (12.0 + 1.0);
        double alpha26 = 2.0 / (26.0 + 1.0);

        ema12[0] = Candles[0].Close;
        ema26[0] = Candles[0].Close;
        for (int i = 1; i < Candles.Count; i++)
        {
            ema12[i] = alpha12 * Candles[i].Close + (1.0 - alpha12) * ema12[i - 1];
            ema26[i] = alpha26 * Candles[i].Close + (1.0 - alpha26) * ema26[i - 1];
        }

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
            
            // EMA distances
            continuous[0, i, 5] = (float)((candle.Close - ema12[i]) / (ema12[i] + 1e-8));
            continuous[0, i, 6] = (float)((candle.Close - ema26[i]) / (ema26[i] + 1e-8));
        }

        List<double> predictedPrices = new();
        double currentClose = Candles.Last().Close;
        DateTime currentTime = Candles.Last().Timestamp;
        double currentEma12 = ema12.Last();
        double currentEma26 = ema26.Last();
        double directionalBias = 0.5;

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

            // Calculate probabilities over vocab via Softmax
            double sumExp = 0.0;
            double[] probs = new double[ModelConfig.VocabSize];
            for (int i = 0; i < ModelConfig.VocabSize; i++)
            {
                float logit = nextTokenLogits[0, ModelConfig.SeqLen - 1, i];
                double expVal = Math.Exp(logit);
                probs[i] = expVal;
                sumExp += expVal;
            }

            double expectedNormReturn = 0.0;
            double probUp = 0.0;
            for (int i = 0; i < ModelConfig.VocabSize; i++)
            {
                probs[i] /= sumExp;
                expectedNormReturn += probs[i] * GetNormReturnFromToken(i);
                if (i >= 128)
                {
                    probUp += probs[i];
                }
            }

            // Expose the directional probability of the very first predicted step
            if (step == 0)
            {
                directionalBias = probUp;
            }

            double predictedLogReturn = expectedNormReturn * rollingVol;
            double nextClose = currentClose * Math.Exp(predictedLogReturn);
            predictedPrices.Add(nextClose);

            // Shift inputs left to make room for the new prediction
            for (int i = 0; i < ModelConfig.SeqLen - 1; i++)
            {
                tokens[0, i] = tokens[0, i + 1];
                for (int j = 0; j < ModelConfig.NumContinuousFeatures; j++)
                    continuous[0, i, j] = continuous[0, i + 1, j];
            }

            // Update EMAs for next step recursively
            currentEma12 = alpha12 * nextClose + (1.0 - alpha12) * currentEma12;
            currentEma26 = alpha26 * nextClose + (1.0 - alpha26) * currentEma26;
            double ema12Dist = (nextClose - currentEma12) / (currentEma12 + 1e-8);
            double ema26Dist = (nextClose - currentEma26) / (currentEma26 + 1e-8);

            // Append new state at sequence end (5-minute bars)
            currentTime = currentTime.AddMinutes(5);
            tokens[0, ModelConfig.SeqLen - 1] = GetToken(expectedNormReturn);
            continuous[0, ModelConfig.SeqLen - 1, 0] = (float)Math.Sin(2 * Math.PI * currentTime.Hour / 24.0);
            continuous[0, ModelConfig.SeqLen - 1, 1] = (float)Math.Cos(2 * Math.PI * currentTime.Hour / 24.0);
            continuous[0, ModelConfig.SeqLen - 1, 2] = (float)Math.Sin(2 * Math.PI * (int)currentTime.DayOfWeek / 7.0);
            continuous[0, ModelConfig.SeqLen - 1, 3] = (float)Math.Cos(2 * Math.PI * (int)currentTime.DayOfWeek / 7.0);
            continuous[0, ModelConfig.SeqLen - 1, 4] = (float)expectedNormReturn;
            continuous[0, ModelConfig.SeqLen - 1, 5] = (float)ema12Dist;
            continuous[0, ModelConfig.SeqLen - 1, 6] = (float)ema26Dist;

            currentClose = nextClose;
        }

        return new PredictionResponse
        {
            PredictedClosePrices = predictedPrices,
            DirectionalBias = directionalBias
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
