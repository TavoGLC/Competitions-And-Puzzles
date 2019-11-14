function find_min_max(number_list){

    let local_min=number_list[0];
    let local_max=number_list[0];

    for(var k=0;k<number_list.length;k++){

        if(number_list[k]<local_min){
            local_min=number_list[k];
        }
        else{
            if(number_list[k]>local_max){
                local_max=number_list[k];
            }
        }
    }
    return [local_min,local_max]
}

console.log(find_min_max([1,4,2,7,6]))
