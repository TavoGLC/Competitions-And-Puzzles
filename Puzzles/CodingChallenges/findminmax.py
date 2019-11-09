#Find the minimum and maximum of a list

def find_min_max(number_list):
    
    list_length=len(number_list)
    local_min=number_list[0]
    local_max=number_list[0]
    
    for k in range(list_length):
        
        if number_list[k]<local_min:
            local_min=number_list[k]
            
        else:
                
            if number_list[k]>local_max:
                local_max=number_list[k]
    
    return local_min,local_max
