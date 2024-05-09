import datetime
import llama_cpp
import gradio as gr


class TextGenerateForever():
    def __init__(self):
        self.prompt = ""
        self.result = ""
        self.buffer = ""
        self.log_file = "log.txt"
        self.max_tokens = 512
        self.temperature = 0.8
        self.model = None
        self.model_info = "No Loaded"
        self.model_path = ""
        self.n_gpu_layers = -1
        self.n_ctx = 0
        self.n_batch = 512
        self.ui().launch()

    def ui(self):
        # gradio components
        with gr.Blocks() as block:
            with gr.Tab(label="Generate") as tab_generate:
                button_generate = gr.Button(
                    value="Generate Forever",
                    variant="primary",
                    visible=True,
                )
                button_stop = gr.Button(
                    value="Stop",
                    variant="stop",
                    visible=False,
                )
                with gr.Group():
                    with gr.Row():
                        textarea_prompt = gr.TextArea(
                            value=lambda: self.prompt,
                            label="Prompt",
                        )
                        textarea_result = gr.TextArea(
                            value=lambda: self.result,
                            label="Result",
                        )
                    textarea_buffer = gr.TextArea(
                        value=lambda: self.buffer,
                        label="Buffer",
                        max_lines=10,
                    )
            with gr.Tab(label="Load Model / Settings") as tab_settings:
                with gr.Group():
                    textbox_log_file = gr.Textbox(
                        value=lambda: self.log_file,
                        label="Log File",
                    )
                    slider_max_tokens = gr.Slider(
                        minimum=0,
                        maximum=4096,
                        value=lambda: self.max_tokens,
                        step=32,
                        label="max_tokens",
                    )
                    slider_temperature = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=lambda: self.temperature,
                        step=0.1,
                        label="temperature",
                    )
                with gr.Group():
                    textbox_model_info = gr.Textbox(
                        value=lambda: self.model_info,
                        label="Model Info",
                    )
                    textbox_model_path = gr.Textbox(
                        value=lambda: self.model_path,
                        label="Model Path",
                    )
                    slider_n_gpu_layers = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=lambda: self.n_gpu_layers,
                        step=0.1,
                        label="n_gpu_layers",
                    )
                    slider_n_ctx = gr.Slider(
                        minimum=0,
                        maximum=32768,
                        value=lambda: self.n_ctx,
                        step=8,
                        label="n_ctx",
                    )
                    slider_n_batch = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=lambda: self.n_batch,
                        step=1,
                        label="n_batch",
                    )
                    button_load = gr.Button(
                        value="Load",
                        variant="primary",
                    )

            # event listeners
            event = button_generate.click(
                fn=self.generate,
                inputs=[
                    textarea_prompt,
                    textarea_result,
                    textarea_buffer,
                ],
                outputs=[
                    button_generate,
                    button_stop,
                    textarea_prompt,
                    textarea_result,
                    textarea_buffer,
                ],
            )

            def click_stop():
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    self.prompt,
                    self.result,
                    self.buffer
                )
            button_stop.click(
                fn=click_stop,
                outputs=[
                    button_generate,
                    button_stop,
                    textarea_prompt,
                    textarea_result,
                    textarea_buffer,
                ],
                cancels=event
            )

            def change_log_file(value):
                self.log_file = value
            textbox_log_file.change(
                fn=change_log_file,
                inputs=textbox_log_file,
            )

            def change_max_tokens(value):
                self.max_tokens = value
            slider_max_tokens.change(
                fn=change_max_tokens,
                inputs=slider_max_tokens,
            )

            def change_temperature(value):
                self.temperature = value
            slider_temperature.change(
                fn=change_temperature,
                inputs=slider_temperature,
            )

            def change_n_gpu_layers(value):
                self.n_gpu_layers = value
            slider_n_gpu_layers.change(
                fn=change_n_gpu_layers,
                inputs=slider_n_gpu_layers,
            )

            def change_n_ctx(value):
                self.n_ctx = value
            slider_max_tokens.change(
                fn=change_n_ctx,
                inputs=slider_n_ctx,
            )

            def change_n_batch(value):
                self.n_batch = value
            slider_max_tokens.change(
                fn=change_n_batch,
                inputs=slider_n_batch,
            )

            button_load.click(
                fn=self.load_model,
                inputs=textbox_model_path,
                outputs=textbox_model_info,
            )
        return block

    def load_model(self, model_path):
        if isinstance(self.model, llama_cpp.Llama):
            if self.model.model_path == model_path:
                self.model_info = f"Aleady Loaded: {model_path}"
                return self.model_info
        try:
            self.model = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_batch=self.n_batch,
            )
            self.model_info = f"Loaded: {self.model.model_path}"
            return self.model_info
        except Exception as e:
            self.model_info = f"Exception: {e}"
            return self.model_info

    def generate(self, prompt, result, buffer):
        log_file = self.log_file
        self.prompt = prompt
        self.result = result
        self.buffer = buffer
        if self.model:
            while True:
                if self.buffer == "":
                    self.buffer = self.prompt
                for stream in self.model(
                    prompt=self.buffer,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    seed=-1,
                    stream=True,
                ):
                    self.buffer += stream["choices"][0]["text"]
                    yield (
                        gr.update(visible=False),
                        gr.update(visible=True),
                        self.prompt,
                        self.result,
                        self.buffer,
                    )
                now = datetime.datetime.now()
                header = f"### {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                buffer = f"{self.buffer}\n\n"
                content = header + buffer
                self.result += content
                self.buffer = ""
                if log_file:
                    with open(log_file, mode="a", encoding="utf-8") as f:
                        f.write(content)
        else:
            yield (
                gr.update(visible=True),
                gr.update(visible=False),
                self.prompt,
                self.result,
                self.buffer,
            )


if __name__ == "__main__":
    TextGenerateForever()
