from abc import ABC, abstractmethod

from utils import InputFeatures, InputExample

from pvp import AgnewsPVP, MnliPVP, YelpPolarityPVP, YelpFullPVP, \
    YahooPVP, PVP, XStancePVP, _prepare ### NEW (_prepare)

PVPS = {
    'agnews': AgnewsPVP,
    'mnli': MnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'xstance': XStancePVP,
    'xstance-de': XStancePVP,
    'xstance-fr': XStancePVP,
}


class Preprocessor(ABC):

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_file: str = None):
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_file) # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}
        ### NEW ###
        self.few_shot_data = None
        ### NEW

    @abstractmethod
    def get_input_features(self, example: InputExample, labelled: bool, **kwargs) -> InputFeatures:
        pass


class MLMPreprocessor(Preprocessor):

    def get_input_features(self, example: InputExample, labelled: bool, **kwargs) -> InputFeatures:
        ### NEW ###
        if self.few_shot_data is not None:
            cls_id = self.wrapper.tokenizer.cls_token_id
            sep_id = self.wrapper.tokenizer.sep_token_id
            mask_id = self.wrapper.tokenizer.mask_token_id

            def preprocessed_ex_ids(ex, labelize):
                ex_input_ids = self.pvp.encode(ex)[0]
                # Remove start (cls) token <s>
                assert ex_input_ids[0] == cls_id
                ex_input_ids = ex_input_ids[1:]
                # Remove double separator </s></s> (if it is there)
                for i in range(len(ex_input_ids)-1):
                    if (ex_input_ids[i:i+2] == [sep_id] * 2):
                        ex_input_ids.pop(i+1)
                        ex_input_ids.pop(i)
                        break
                if not labelize: return ex_input_ids
                # Replace <mask> with the label
                label = _prepare(self.pvp.verbalize(ex.label)[0],
                                 self.wrapper.tokenizer)
                label_id = self.wrapper.tokenizer.convert_tokens_to_ids(label)
                return [label_id if tok_id == mask_id else tok_id
                        for tok_id in ex_input_ids]

            input_ids = preprocessed_ex_ids(example, labelize=False)
            cond, num_cond = [], 0
            for ex in self.few_shot_data:
                new_ex = preprocessed_ex_ids(ex, labelize=True)
                if (1 + len(cond) + len(new_ex) + len(input_ids) >
                    self.wrapper.config.max_seq_length): break
                cond.extend(new_ex)
                num_cond += 1
            input_ids = [cls_id] + cond + input_ids
            token_type_ids = [0] * len(input_ids)

            # print(f'Conditioning on {num_cond}/'
            #       f'{len(self.few_shot_data)} examples; '
            #       f'labels: {[e.label for e in self.few_shot_data[:num_cond]]}')
        else:
            input_ids, token_type_ids = self.pvp.encode(example)
        ### NEW ###

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        ### NEW ###
        # print('*****************************')
        # print(self.wrapper.tokenizer.decode(input_ids))
        # assert False
        ### NEW ###

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label]
        logits = example.logits if example.logits else [-1]

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits)


class SequenceClassifierPreprocessor(Preprocessor):

    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:
        inputs = self.wrapper.tokenizer.encode_plus(
            example.text_a if example.text_a else None,
            example.text_b if example.text_b else None,
            add_special_tokens=True,
            max_length=self.wrapper.config.max_seq_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs.get("token_type_ids")

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        if not token_type_ids:
            token_type_ids = [0] * self.wrapper.config.max_seq_length
        else:
            token_type_ids = token_type_ids + ([0] * padding_length)
        mlm_labels = [-1] * len(input_ids)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label]
        logits = example.logits if example.logits else [-1]

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             label=label, mlm_labels=mlm_labels, logits=logits)
