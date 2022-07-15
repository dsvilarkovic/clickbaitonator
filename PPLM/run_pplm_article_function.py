from run_pplm_article import run_pplm_example
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

article_content = """Australian actor Guy Pearce will return for the iconic soap Neighbours finale on August 1 to reprise his role as Mike Young.
                    Guy, 54, played the troubled Mike from 1986 to 1989, and is now set to make a comeback on the show after 33 years, Metro.co.uk reports.
                    The star's character arcs explored the implications of domestic abuse, student-teacher relationships and dealing with loss of loved ones.
                    Speaking to Metro.co.uk, Guy said: 'It is very exciting and surreal at the same time being back on set again, however it feels like coming home.
                    'It's where it all started for me professionally. I've been asked to come back on occasions over the years and wondered if it was the right thing 
                    to do, but once I knew the show was finishing, I knew I had to do it.'He added that there is 'nothing like being here all together again'
                    , even though he's had a chance to catch-up with other cast members."""

num_samples = 5
num_iterations = 5


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


run_pplm_example(   discrim="clickbait_mpnet", 
                    model=model,
                    tokenizer=tokenizer,
                        pretrained_model="google/pegasus-xsum",
                        uncond=True,
                        num_samples=num_samples,
                        class_label=1,
                        length=50,
                        gamma=1.0,
                        num_iterations=num_iterations,
                        stepsize=0.04,
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
                        verbosity="very_verbose",
                        article_content=article_content,
                    )