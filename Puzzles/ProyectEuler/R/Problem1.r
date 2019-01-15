
end<-999
container<-0

for (i in 0:end){
  
  cVal=i
  
  if(cVal%%3==0 | cVal%%5==0){
    
    container<-container+cVal
    
  } 
  
}
