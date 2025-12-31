import torch

a = torch.tensor([[3,1,8], 
                  [0,9,2]])

print("============storage==========")
print("Storage:\n", a.storage())
print("Offset: ", a.storage_offset())
print("Strides: ", a.stride())
print("Order of elements in a:")
for x in a.flatten():
    print(x.item())
print("=============================")
print()

print("============memory===========")
print("One element takes", a[0,1].element_size(), "bytes")

for x in a.flatten():
    print(f"Memory adress of {x.item()}: {x.data_ptr()}")
print("=============================")
print()

print("======example: transpose=====")
a_t = a.T
print("is a.T contiguous:", a_t.is_contiguous())
print("Storage of a.T:\n", a_t.storage())
print()

print("Order of elements in a.T:")
for x in a_t.flatten():
    print(x.item())

a_t_contiguous = a_t.contiguous()
print("is a.T.contiguous contiguous:", a_t_contiguous.is_contiguous())
print("Storage of a.T.contiguous:\n", a_t_contiguous.storage())
print()

print("Order of elements in a.T.contiguous():")
for x in a_t_contiguous.flatten():
    print(x.item())
print("=============================")
print()
print()
print()


print("========example: views=======")
a_view = a.view(6,1)

print("a_view using .view():",a_view)
print("Strides: ", a_view.stride())

print("is a_view contiguous:", a_view.is_contiguous())
print("Storage of a_view:\n", a_view.storage())
print()

print("Order of elements in a_view:")
for x in a_view.flatten():
    print(x.item())
