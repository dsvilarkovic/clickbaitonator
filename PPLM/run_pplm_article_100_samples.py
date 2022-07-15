from run_pplm_article import run_pplm_example
# import run_pplm_example
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from datasets import load_dataset,DatasetDict,Dataset
# from datasets import 
# from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from run_pplm_article import run_pplm_example
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
import pandas as pd

device='cuda'
webis_train = "https://ml-coding-test.s3.eu-west-1.amazonaws.com/webis_train.csv"
webis_test = "https://ml-coding-test.s3.eu-west-1.amazonaws.com/webis_test.csv"
df_train = pd.read_csv(webis_train)
df_test = pd.read_csv(webis_test)


df_train['truthClass'] = pd.factorize(df_train['truthClass'])[0]
df_test['truthClass'] = pd.factorize(df_test['truthClass'])[0]

df_train, df_valid = train_test_split(df_train, test_size = 0.2, random_state = 42, 
                                  stratify = df_train['truthClass'])
num_samples = 2
num_iterations = 2


pretrained_model = "google/pegasus-xsum"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

tokenizer.add_special_tokens({'pad_token': '<pad>'})

model = AutoModelForSeq2SeqLM.from_pretrained(
    pretrained_model,
    output_hidden_states=True
)

model.to(device)
model.eval()

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False


def create_clickbait_samples(df_set, 
                            device='cpu',
                            num_samples = 5,
                            num_iterations = 5,
                            save_file=None,
                            stepsize=0.04,
                            subset = 100):
    """
    Generate clickbait samples from a dataset.
    """

    pretrained_model = "google/pegasus-xsum"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False


    df_clickbait_set = pd.DataFrame({}, columns=['original_title', 'clickbait_title', 'article_content'])

    for row in tqdm(df_set.iloc[0:subset].iterrows()):
        print('__' * 20)
        article_content = row[1]['targetParagraphs']
        # print(row[0]['targetTitle'])
        results = run_pplm_example(   discrim="clickbait_mpnet", 
                        model = model,
                        tokenizer = tokenizer,
                        pretrained_model="google/pegasus-xsum",
                        uncond=True,
                        num_samples=num_samples,
                        class_label=1,
                        length=50,
                        gamma=1.0,
                        num_iterations=num_iterations,
                        # stepsize=0.04,
                        stepsize=stepsize,
                        kl_scale=0.01,
                        gm_scale=0.95,
                        sample=True,
                        temperature=1.0,
                        top_k=10,
                        grad_length=10000,
                        window_length=0,
                        horizon_length=1,
                        decay=False,
                        seed=0,
                        no_cuda=False,
                        colorama=False,
                        verbosity="quiet",
                        article_content=article_content[0:2000],
                    )

        original_title = results['unperturbed'].replace('</s>', '').replace('<pad>', '')
        # clickbait_title = results['unperturbed'].replace('</s>', '').replace('<pad>', '')
        result_row = {
            'original_title' : original_title,
            'article_content' : article_content,
        }

        result_row.update(results) #added results 

        # for i,generated_text in enumerate(generated_texts):
          # result_row = [f'perturbed_{i}']

        print('\n')
        # print(f'Original title: {original_title}')
        print(result_row)
        print('\n')        
        # print(f'Clickbait title:  {clickbait_title}')
        df_clickbait_set = df_clickbait_set.append(result_row, ignore_index=True)

    df_clickbait_set.to_csv(save_file, index=False)
    return df_clickbait_set


stepsize_array = [0.4, 0.8, 1.2, 1.6, 2.0]

device='cuda'
for stepsize in stepsize_array:
    df_stepsize_result = create_clickbait_samples(df_test, 
                                device=device,
                                save_file='df_clickbait_test',
                                num_samples = 5,
                                num_iterations = 5,
                                subset=25,
                                stepsize=stepsize)
    df_stepsize_result.to_csv('../drive/MyDrive/nlp_lss_data/df_clickbait_test_stepsize_' + str(stepsize) + '.csv', index=False)
