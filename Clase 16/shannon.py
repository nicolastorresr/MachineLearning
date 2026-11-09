#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import pipeline


# In[ ]:


from transformers import logging

logging.set_verbosity_error()


# In[ ]:


BERT = pipeline('fill-mask', model='bert-base-uncased')


# In[ ]:


GPT2 = pipeline('text-generation', model='gpt2')


# In[ ]:


def highlight(colour, text):
    if colour == "red":
        return "\033[0;41m" + str(text) + "\033[0;m"
    if colour == "blue":
        return "\033[0;44m" + str(text) + "\033[0;m"
    return str(text)


# In[ ]:


def prediction(text):
    text = text.strip()
    bert = BERT(f'{text} [MASK].')
    bert_output = bert[0]['token_str']    
    gpt2 = GPT2(f'{text}', num_return_sequences=1, 
                     max_new_tokens=1, pad_token_id=GPT2.tokenizer.eos_token_id)
    gpt2_output = gpt2[0]['generated_text'].split(" ")[-1]
    if gpt2_output.startswith(text):
        gpt2_output = gpt2_output.replace(text,"")
        gpt2_output = gpt2_output.strip() 
    return bert_output, gpt2_output


# In[ ]:


def evaluate(next_word, guess, text):
    guess = guess.replace('.','')
    guess = guess.replace(',','')
    next_word = next_word.replace('.','')
    next_word = next_word.replace(',','')
    if guess == next_word:
        return(text + " " + highlight("blue",guess))
    else:
        return(text + " " + highlight("red",guess))


# In[ ]:


def score(next_word, guess):
    guess = guess.replace('.','')
    guess = guess.replace(',','')
    next_word = next_word.replace('.','')
    next_word = next_word.replace(',','')
    if guess == next_word:
        return 1
    else:
        return 0


# In[ ]:


def shannon(full):
    full = full.split()
    text = ''
    bert_acc = gpt2_acc = 0
    bert = full[0] 
    gpt2 = full[0]
    for i in range(len(full)-1):
        word = full[i]
        next_word = full[i+1]
        text = text + word + " "
        bert_guess,gpt2_guess = prediction(text)
        bert = evaluate(next_word,bert_guess,bert)
        gpt2 = evaluate(next_word,gpt2_guess,gpt2)
        bert_acc += score(next_word,bert_guess)
        gpt2_acc += score(next_word,gpt2_guess)
    text = text + full[i+1]   
    print(f"BERT: {bert} ({round(((bert_acc*100)/(len(full)-1)))}%)")
    print(f"GPT2: {gpt2} ({round(((gpt2_acc*100)/(len(full)-1)))}%)")

