using Augurus.Api;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Initialize the ONNX Inference Engine as a Singleton
string onnxPath = Path.Combine(builder.Environment.ContentRootPath, "..", "financial_transformer.onnx");
string quantizerPath = Path.Combine(builder.Environment.ContentRootPath, "..", "quantizer_bins.json");

if (!File.Exists(onnxPath) || !File.Exists(quantizerPath))
{
    Console.WriteLine("CRITICAL WARNING: ONNX Model or Quantizer JSON not found in inference directory!");
}
else 
{
    var engine = new InferenceEngine(onnxPath, quantizerPath);
    builder.Services.AddSingleton(engine);
}

// Enable CORS for OpenBackTest UI
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", builder =>
        builder.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();
app.UseCors("AllowAll");

app.MapPost("/api/predict", ([FromBody] PredictionRequest request, InferenceEngine engine) =>
{
    try
    {
        var response = engine.Predict(request.Candles);
        return Results.Ok(response);
    }
    catch (Exception ex)
    {
        return Results.BadRequest(new { error = ex.Message + "stacktrace:" + ex.StackTrace });
    }
})
.WithName("PredictMarketMove")
.WithOpenApi();

app.Run();
