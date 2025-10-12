from vllm import LLM

if __name__ == "__main__":
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate("The capital city of Roman empire is")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


