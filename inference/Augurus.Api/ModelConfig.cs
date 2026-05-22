namespace Augurus.Api;

public static class ModelConfig
{
    // These parameters must match the Python training configuration exactly.
    // They dictate the shape of the tensors fed into the ONNX model.
    
    public const int SeqLen = 64;
    public const int VocabSize = 256;
    public const int NumContinuousFeatures = 7;
    
    // Architectural parameters (Baked into ONNX, kept here for reference)
    public const int DModel = 128;
    public const int NHead = 8;
    public const int NumLayers = 5;
}
