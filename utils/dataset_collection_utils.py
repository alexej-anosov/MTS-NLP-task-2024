import sys

def get_step_score(train_df, input_text, output):
    
    step = f'INPUT\n{input_text}\nOUTPUT{output}\n\n'
    print(step)
    
    if input('Send 1 if the step is correct, else 0: ') == '1':
        train_df.loc[len(train_df)] = [input_text, output]
            
    return train_df