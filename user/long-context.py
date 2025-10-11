from vllm import LLM, SamplingParams
import sys

if __name__ == "__main__":

    llm = LLM(model="facebook/opt-125m")

    input_file = sys.argv[1]

    long_input =  [open(input_file).read(), ]


    outputs = llm.generate(long_input)

    for line in outputs:
        print(f" >>> output line :{line}")




