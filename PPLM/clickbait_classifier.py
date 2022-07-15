import torch
from transformers import BertModel, BertConfig, PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import TokenClassifierOutput,SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
import torch.nn as nn
# from modeling_mpnet import MPNetModel, MPnetConfig

class ClickbaitConfig(PretrainedConfig):
    def __init__(
        self,
        model_type: str = "bert",
        pretrained_model: str = "bert-base-uncased",
        num_labels: int = 1,
        dropout: float = 0.1,
        inner_dim1: int = 256,
        inner_dim2: int = 32, 
        max_length: int = 512,
        load_pretrained: bool = True,
        freeze_bert: bool = True,
        **kwargs
    ):
        super(ClickbaitConfig, self).__init__(num_labels=num_labels, **kwargs)
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        self.inner_dim1 = inner_dim1
        self.inner_dim2 = inner_dim2
        self.max_length = max_length
        self.load_pretrained = load_pretrained
        self.freeze_bert = freeze_bert


class BertClickbaitClassifier(PreTrainedModel):
    """
      Taken and extended from BertforSequenceClassification : https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/bert/modeling_bert.py#L1508
    """
    config_class = ClickbaitConfig
    def __init__(self, config: ClickbaitConfig):
        super(BertClickbaitClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.bert_config = BertConfig.from_pretrained(config.pretrained_model)
        self.bert_config = AutoConfig.from_pretrained(config.pretrained_model)

        # self.bert = BertModel(self.bert_config)
        self.bert = AutoModel.from_pretrained(config.pretrained_model, config=self.bert_config)
        # self.bert = SentenceTransformer(config.pretrained_model, config=self.bert_config)
        # self.bert = MPNetModel(config.pretrained_model, config=self.bert_config)
        if config.load_pretrained:
            print("Load pretrained weights from {}".format(config.pretrained_model))
            self.bert = self.bert.from_pretrained(config.pretrained_model)
        if config.freeze_bert:
            print("Freeze weights in the BERT model. Just the classifier will be trained")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.linear_1 = nn.Linear(self.bert.config.hidden_size, config.inner_dim1)
        self.dropout_1 = nn.Dropout(config.dropout) 
        self.relu_1 = nn.ReLU()
        self.dropout_2 = nn.Dropout(config.dropout)
        self.linear_2 = nn.Linear(config.inner_dim1, config.inner_dim2)
        self.relu_2 = nn.ReLU()
        self.dropout_3 = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.inner_dim2, config.num_labels)
        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = outputs[0][:,0,:]

        x = self.dropout_1(output)
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        x = self.dropout_3(x)

        logits = self.classifier(x)
        logits = self.sigmoid(logits)

        loss = None
        if labels is not None:
            # loss_fct = BCELoss(weight=WEIGHT)
            loss_fct = BCELoss()
            labels = 1.0*labels
            loss = loss_fct(logits.view(-1), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )