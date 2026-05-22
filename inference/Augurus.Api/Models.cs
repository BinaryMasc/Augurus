namespace Augurus.Api;

public class PredictionRequest
{
    // The frontend should send exactly the number of candles defined in ModelConfig.SeqLen
    public List<Candle> Candles { get; set; } = new();
}

public class Candle
{
    public DateTime Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
}

public class PredictionResponse
{
    public List<double> PredictedClosePrices { get; set; } = new();
    public double DirectionalBias { get; set; } // Probability of next candle being Up (0.0 to 1.0)
}
