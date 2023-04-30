
name = "NJ"
words = "hello"

print("%s said %s"%(name, words))
print('{} said {}'.format(name, words))
print('{1} kata {0}'.format(name, words))
print(f'{name} said {words}')


phi = 3.1428
print("{:2f}".format(phi))
print("%.2f"%(phi))
print(f"{phi:.2f}")

# LIST
my_list = []
print(type(my_list))

my_list.append([1,2,3])
my_list.append([4,5,6])

print(my_list)


for i in my_list:
    for j in i:
        print(j)

my_tuple = ('a', 'b', 'c')
print(my_tuple)
q, w, e = my_tuple
print(q, w, e)

my_set = {1, 2, 2, 1, 2, 3, 4, 2, 1}
print(my_set)

my_dict = {
    'key1' : 2,
    'key2' : 8,
    'key3' : 10
}

print(my_dict['key3'])
print(my_dict.items())
