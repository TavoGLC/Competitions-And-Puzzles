
def disemvowel(text):
    
    text = list(text)
    new_letters = []
    last_vowel_state=False
    
    for i in text:
        if i.lower() == "a" or i.lower() == "e" or i.lower() == "i" or i.lower() == "o" or i.lower() == "u":
            last_vowel_state=True
            pass
        else:
            if last_vowel_state==True and i.lower()=='g':
                pass
            else:    
                new_letters.append(i)
            last_vowel_state=False
            

    print (''.join(new_letters))
