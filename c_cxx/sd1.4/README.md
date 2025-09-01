## Stable Diffusion v1.4 C++ sample
## Build Instructions
* Build ORT shared lib with NV EP 
* From ORT directory, install using `cmake --install .\<build folder>\Release --config Release` (Installs to C:/Program Files/onnxruntime by default)
* Change directory to onnxruntime-inference-examples/c_cxx/sd1.4
* `mkdir build && cd build`
* Generate build files: `cmake .. -A x64 -T host=x64 -DONNXRUNTIME_ROOT_PATH="C:/Program Files/onnxruntime" "-Donnxruntime_USE_NV=ON"`
* Open .sln in Visual Studio and build sd_pipeline project

## Execution
* Copy `cudart64_12.dll, onnxruntime.dll, onnxruntime_providers_nv_tensorrt_rtx.dll, onnxruntime_providers_shared.dll, ortextensions.dll` and TRT-RTX dlls to the build folder
* Setup SD 1.4 onnx model folder with clip tokenizer model
* Run using `sd_pipeline.exe --model_path "D:\repos\stable-diffusion-v1-4" --prompt "a rock concert" --output output.png`