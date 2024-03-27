import os
import re
import json
import time
import openai
from rich import pretty
from tqdm import tqdm
pretty.install()
from generate_alpaca_data import num_tokens_from_messages

openai.api_type = "azure"
openai.api_base = "https://senzgpt.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "68a0be7e50f04895a0a99e997b73f471"

GENERATE_JSON_INSTRUCTION_PROMPT = """Suppose you are an algorithmic engineer who specializes in producing instruction datasets, and we are now collaborating on producing some instruction datasets organized from the literature. The format of the dataset is as follows:
{
"instructions": xxx
"output": xxx
}
The "output" of the dataset is a passage from the literature, while the "instructions" need to be refined by you.
I will give an output below and you need to complete the following tasks:
1. Please give the corresponding instructions according to the output, it can be a simple question or a short instruction, so that the AI can answer the corresponding output after accepting the instruction. don't have similar meanings of the instructions in the generated instructions. It is better to have only one or two sentences, which can be contextualized with the output. 
2. please modify the original output to make it smoother, more logical, easier to understand, more engaging, and more insightful, while retaining the details and removing information such as Fig, Figure, Table, and Citation Format [%d]. 
3. please output the instructions and modified output in json format.
Now I will give you an output where you output a dataset in json format and do not need to reply with anything else."""

GENERATE_OUTPUT_PROMPT = """Assuming you are an editor who is particularly good at rewriting and embellishing, let's now edit some text together.
Below I give you a passage extracted from the paper, and you need to complete the following tasks:
1. adapt the passage from the original text into a perfect answer for the AI. It is important to retain as much detail as possible from the original text, e.g. units, quantifiers, etc.
2. delete words such as this thesis, this study, and information such as Fig, Figure, Table, and citation format [%d] from the response
3. optimize the content of the answer to make the language more fluent and easier to understand while retaining the original meaning.
Below I will give you a passage, please output the modified answer directly, no need to output other content:"""

GENERATE_INSTRUCTIONS_PROMPT = """Suppose you are an algorithmic engineer who specializes in producing instruction datasets, and we are now collaborating on producing some instruction datasets organized from the literature. The format of the dataset is as follows:
{
"instructions": xxx
"output": xxx
}
The "instructions" in this dataset are the human inputs, which can be questions or instructions, and the "outputs" are the AI's answers.
I'm going to give you an "output", and you need to do the following:
1. find the best "instructions" that match the content of the "output", so that the AI can answer the "output" without contextualization; 
2. "instructions" should preferably be one or two sentences, either questions or instructions, that correspond to the core content of the "output". Do not include words like this paper, this study, etc.
Below I will give you an "output", please output an "instructions" that satisfies the task requirements, note that you output the "instructions" directly, no other content is needed."""

DAVINCI_PROMPT = """Suppose you are an algorithm engineer who specializes in making instruction datasets, now let's make instruction datasets together. The example dataset is as follows:
  {
        "Instructions": xxx,
        "output": xxx.
 }
Below I will give you a paragraph and you need to accomplish the following tasks:
1. extract the most important elements of the paragraph without changing its meaning. Be careful to retain specific details such as units, quantifiers, etc.
2. Adapt the extracted content in 1 into an answer to a question and optimize the presentation to make it more concise, organized, fluent, easy to understand, while preserving the original meaning and specific details. Use the adapted answer as the "output" of the dataset.
3. For the answers generated in step 2, design a matching question or description to serve as the "instructions" of the dataset. Please note that the "instructions" should be concise and the words "Research" and "Paper" should not appear in the "instructions".
4. Combine the "output" from 2 and the "instructions" from 3 into a format that describes the dataset and output it.
I will give a paragraph below, please complete the task as required. Just output the final instruction dataset and output it in json format without any other action:
Paragraph:
%s"""

SCORE_PROMPT = """Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user’s instruction. Please assign a score using the following 5-point scale:
1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user’s question. Or the response is from another person’s perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.
2: It means the answer addresses most of the asks from the user. It does not directly address the user’s question. For example, it only provides a high-level methodology instead of the exact solution to user’s question.
3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant’s perspective, but from other people’s perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.
4: It means the answer is written from an AI assistant’s perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user’s question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.
5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user’s question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful.
Please first provide a brief reasoning you used to derive the rating score, and then write "Score: " in the last line.
%s"""

def assistant_message(content):
    return {
        'role': "assistant",
        'content': content
    }

def human_message(content):
    return {
        'role': "user",
        'content': content
    }

def extract_ai_message(m):
    return {
        'role': "assistant",
        'content': m.message.content
    }

class LLM:
    def __init__(self) -> None:
        self.token = 0
        self.cur_info = 0

    def compute_token(self, response):
        total = response.usage["total_tokens"]
        self.token += total
        if t := self.token // 1000 > self.cur_info:
            self.cur_info = t
            print(f"current token: {self.token}")


class DAVINCI(LLM):
    def __init__(self) -> None:
        super().__init__()

    def _api(self, input):
        prompt = DAVINCI_PROMPT % input
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.compute_token(response)
        time.sleep(2)
        return response.choices[0]["text"]

    def generate(self, input):
        return extract_json_data(self._api(input))


def extract_json_data(contents: str):
    contents = re.sub("\[.*?\]", "", contents)
    contents = re.sub("Instructions", "instructions", contents)
    pattern = r'\{(?s)(.*)\}'
    match = re.search(pattern, contents)
    if match:
        data = json.loads(match.group())
    instruction = data["instructions"]
    if "study" in instruction or "paper" in instruction or "article" in instruction or "literatures" in instruction:
        data = {}
    return data


class GPT(LLM):
    def __init__(self):
        super().__init__()

    # def compute_token(self, response):
    #     total = response.usage["total_tokens"]
    #     self.token += total
    #     if t := self.token // 1000 > self.cur_info:
    #         self.cur_info = t
    #         print(f"current token: {self.token}")

    def _api(self, messages):
        time.sleep(1)
        num = num_tokens_from_messages(messages)
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            engine="TestGPT",
            messages=messages,
            temperature=0,
            max_tokens=4097-num,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        self.compute_token(response)
        return response
    
    def generate_score(self, inputs):
        messages = []
        messages.append(human_message(SCORE_PROMPT % inputs))
        return messages
    
    def generate_output(self, inputs):
        messages = []
        messages.append(human_message(GENERATE_OUTPUT_PROMPT))
        messages.append(assistant_message("Sure, I'd be happy to help you with that! Please provide me with the passage you would like me to edit."))
        messages.append(human_message(inputs))
        return messages
    
    def generate_instruction(self, inputs):
        messages = []
        messages.append(human_message(GENERATE_INSTRUCTIONS_PROMPT))
        messages.append(assistant_message("What is the 'output' that you would like me to work with?"))
        messages.append(human_message(inputs))
        return messages
    
    def create_generate_json_directly_message(self, inputs):
        messages = []
        messages.append(human_message(INSTRUCTION_PROMPT))
        # response = self._api(messages)
        # messages.append(extract_ai_message(response.choices[0]))
        ai_message = '{\n  "instructions": "What are the key factors that contribute to the success of a business?",\n  "output": "The success of a business is influenced by various factors, including market demand, effective marketing strategies, strong leadership, competitive advantage, and financial stability. These factors play a crucial role in determining the growth and profitability of a business."\n}'
        messages.append(assistant_message(ai_message))
        messages.append(human_message(inputs))
        return messages
    
    def convert_inputs(self, inputs):
        messages = self.generate_output(inputs)
        response = self._api(messages)
        output = response.choices[0].message.content
        output = re.sub("\[.*?\]", "", output) 
        messages = self.generate_instruction(output)
        response = self._api(messages)
        instruction = response.choices[0].message.content
        return make_alpaca_data(instruction, output)

    def get_alpaca_data(self, inputs):
        messages = []
        messages.append(human_message(DAVINCI_PROMPT % inputs))
        response = self._api(messages)
        output = response.choices[0].message.content
        return extract_json_data(output)

    
    def get_score(self, inputs):
        score = 0
        messages = self.generate_score(inputs)
        response = self._api(messages)
        output = response.choices[0].message.content
        pattern = r'(core:.*)[0-5]'
        match = re.search(pattern, output)
        if match:
            score = int(match.group().split(":")[-1])
        return score
        
    
    # def filter_content(self, inputs):
    #     prompt = concat_prompt(inputs, "filter")
    #     messages = create_message(prompt)
    #     retry, value = 0, False
    #     while retry < 3:
    #         try:
    #             value = self._filter_content(messages)
    #             return value
    #         except Exception as e:
    #             retry+=1
    #     if retry >= 3:
    #         print(f"Can't filter: {inputs}")
    #     else:
    #         return False
    
    # def _filter_content(self, messages):
    #     response = self._api(messages)
    #     content = response.choices[0].message.content
    #     return eval(content)

def make_alpaca_data(instructions, output):
    alpaca_data = {
        "instructions": instructions,
        "output": output
    }
    return alpaca_data

def is_single_word(input):
    li = input.split(" ")
    if len(li) < 4:
        if len(list(filter(None, li))) == 1:
            return True
    return False
# Introduction, Materials and methods, Results, Summary, Discussion, 
def extract_main_content(c):
    record = False
    methods = {}
    cur_key = ""
    li = c.split("\n")
    for l in li:
        if "ntroduction" in l and len(l.split(" ")) < 6:
            cur_key = "introduction"
            methods[cur_key] = []
            record = True
            continue
        if "ethod" in l and len(l.split(" ")) < 6:
            cur_key = "method"
            methods[cur_key] = []
            record = True
            continue
        # if ("esult" in l or "isscuss" in l) and len(l.split(" ")) < 6:
        #     break
        if record:
            if is_single_word(l):
                record = False
            else:
                methods[cur_key].append(l.strip())
    contents = []
    # for key, value in methods.items():
    #     if value:
    #         contents.append("\n".join(value))
    if "introduction" in methods:
        contents.append("\n".join(methods["introduction"]))
    return contents


def extract_method_content(c):
    record = False
    methods = []
    li = c.split("\n")
    for l in li:
        if "ethod" in l:
            record = True
            continue
        if "esult" in l:
            record = True
            continue
        if record:
            if is_single_word(l):
                record = False
            else:
                methods.append(l.strip())
    return methods

class File:
    """
    Read file
    write lines
    write json
    """
    def __init__(self, path) -> None:
        self.path = path

    def read(self) -> str:
        with open(self.path, "r") as f:
            return f.read()
        
    def append_lines(self, content):
        with open(self.path, "a") as f:
            f.write(content)
            f.write("\n")


def read_datasets(base_path):
    # with open(os.path.join(base_path, "done.txt"), "r") as sf:
    #     solve_files = set(sf.read().split("\n"))
    f = File(os.path.join(base_path, "done.txt"))
    solve_files = set(f.read().split("\n"))
    file_list = os.listdir(base_path)
    for file in tqdm(file_list):
        if file in solve_files or ".json" not in file:
            continue
        try:
            body = json.load(open(os.path.join(base_path, file), "r"))["body"]
            contents = extract_main_content(body)
            yield contents
            print(f"solved {file}")
            f.append_lines(file)
        except:
            print(f"error file {file}")


def save_json(content, store_path):
    array = json.load(open(store_path, "r+", encoding="utf-8"))
    array.append(content)
    json.dump(array, open(store_path, "w+", encoding="utf-8"))

# def extract_key_words(base_path):
#     key_words = set()
#     for file in os.listdir(base_path):
#         if ".json" not in file:
#             continue
#         body = json.load(open(os.path.join(base_path, file), "r"))["body"]
#         li = body.split("\n")
#         for l in li:
#             if is_single_word(l):
#                 key_words.add(l.strip().lower())
#     print(key_words)


def extract_files(path="./datas/pmc", store_path="./datas/result/pmc"):
    final_answer, num = [], 0
    store_file_format = "alpaca_data_{}.json"
    gpt = GPT()

    i = 0
    for content in read_datasets(path):
        for c in content:
            try:
                data = gpt.get_alpaca_data(c)
                if not data:
                    continue
                score = gpt.get_score(json.dumps(data))
                if score >= 4:
                    data["score"] = score
                    final_answer.append(data)
                    num += 1
            except Exception as e:
                print(e)
        if num // 2 > i:
            json.dump(final_answer, open(os.path.join(store_path, store_file_format.format(i)), "w+"))
            print(f"save file to {store_file_format.format(i)}")
            final_answer.clear()
            i = num // 2
    if final_answer:
        json.dump(final_answer, open(os.path.join(store_path, store_file_format.format(i+1)), "w+"))
        print(f"save file to {store_file_format.format(i+1)}")



if __name__ == "__main__":
    extract_files("./datas/pmc")
    # test_alpaca_data = json.load(open("./datas/alpaca_data_2.txt", "r"))
    pass
