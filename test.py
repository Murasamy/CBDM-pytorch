# from torch_utils import misc

a = "layer1"
a_list = []
a_list.append(a)
print(a_list)
b_list = a_list.clone()
# check if a in b_list has the same memory address
print(b_list)