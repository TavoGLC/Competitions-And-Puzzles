
nums<-c(3,5,1,2,4,8)

find_min_max<-function(number_list){

  list_length=length(number_list)
  local_min=number_list[1]
  local_max=number_list[1]
  
  for(k in 1:list_length){
  
    if(number_list[k]<local_min){
      local_min=nums[k]
    }
    
    else{
      
      if(number_list[k]>local_max){
      local_max=number_list[k]
      }
    }
  }
  
  output=c(local_min,local_max)
}
