from vllm import LLM

def main():
    llm = LLM(model="facebook/opt-1.3b")
    # llm = LLM(model="Qwen/Qwen2-7B", tensor_parallel_size=2, pipeline_parallel_size=2)
    outputs = llm.generate(
        [
            "The capital city of Roman empire is ",
            "Rome is the capital city of",
            "Constantinople is the capital city of"
        ]
    )

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f" >>>>>> Prompt <<<<<<\n {prompt}")
        print(f" >>>>>> Generated text <<<<<<\n {generated_text}")

    
    
if __name__ == "__main__":
    main()

