class InstructionsHandler:
    def __init__(self):
        self.pi = {}

    def load_instruction_set(self, ):
        # instruction v3
        self.pi['bos_instruct'] = """Instruction: If the given two sentences convey the same meaning, output "Positive"; otherwise, output 'Negative'.
        Positive example -
        Sentence1: How do I recover my Google account? 
        Sentence2: I lost my Google account information. How can I recover it?
        Output: Positive
        Negative example -
        Sentence1: Most innovation Android apps of 2016?
        Sentence2: What are most innovative Android apps of 2015?
        Output: Negative
        Now complete the following example -
        Sentence 1: {} 
        Sentence 2: {} """
        self.pi['eos_instruct'] = '\nOutput: '

        # instruction v2
        # self.pi['bos_instruct'] = "Please identify whether the given two questions are conveying the same meaning. {} {}"
        # self.pi['eos_instruct'] = ''

        # instruction v1
        self.pi['bos_instruct'] = """Instruction: Please identify whether the given two sentences are conveying the same meaning. The answer should be exactly 'Yes' or 'No'.
        Positive example -
        Sentence1: How do I recover my Google account? 
        Sentence2: I lost my Google account information. How can I recover it?
        Output: Yes
        Negative example -
        Sentence1: Most innovation Android apps of 2016?
        Sentence2: What are most innovative Android apps of 2015?
        Output: No
        Now complete the following example -
        Sentence 1: {} 
        Sentence 2: {} """
        self.pi['eos_instruct'] = '\nOutput: '

        self.pi['delim_instruct'] = ''

        # self.ate['bos_instruct1'] = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
        # Positive example 1-
        # input: I charge it at night and skip taking the cord with me because of the good battery life.
        # output: battery life
        # Positive example 2-
        # input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
        # output: features, iChat, Photobooth, garage band
        # Now complete the following example-
        # input: """

        # self.ate['delim_instruct'] = ''
        # self.ate['eos_instruct'] = ' \noutput:'