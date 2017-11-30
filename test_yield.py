def test_yield(ls):
	yield ls

ls = (1,2,3,4,3,2)

m = test_yield(ls)
# yield returns a generator just as net.parameters()
for i in m:
	print i