This is **a Simple Webui for Generating Text** using gradio.  
It was influenced by Zuntan03's EasyNovelAssistant.  
https://github.com/Zuntan03/EasyNovelAssistant  

Installation:  
I recommend setting up a virtual environment.  
Install gradio and llama-cpp-python following their respective official guides.  
https://github.com/gradio-app/gradio  
https://github.com/abetlen/llama-cpp-python  

Usage:  
Run the `text_generate_forever.py` script in the console.
```
python text_generate_forever.py
```
Open the URL displayed.
```
Running on local URL:  http://127.0.0.1:7860
```
1. Go to the `Load Model / Settings` tab and specify the path to load the gguf in `Model Path`.
2. You can also set options like `n_gpu_layers`, `n_ctx`, and `n_batch` here, which are the same as those used in llama-cpp-python.
    - `n_gpu_layers` is the number of layers that will be placed on the GPU memory. If set to -1 to placed all layers onto the GPU memory.
    - `n_ctx` is the length of the context. Longer context lengths make replies smarter, but consume more memory. You can set it to 0 to use gguf's default context length.
    - `n_batch` is the batch size. Setting it to a lower number will consume less memory, but take more time to prepare for inference.
4. Write a prompt in the `Generate` tab and press the `Generate Forever` button.
5. Text accumulates in Buffer. Once `max_tokens` are reached, the content is copied into `Result` and `Buffer` is cleared, until you press the `Stop` button again.
    - You can also change settings like `max_tokens` and `temperature` while generating text. (temperature affects the oddity of generated sentences).
    - The generated buffer is also appended to a `Log File`. If you won't logging, leave it empty.
