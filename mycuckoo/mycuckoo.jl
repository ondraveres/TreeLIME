using Cuckoo
using JsonGrinder
using PrintTypesTersely


PrintTypesTersely.on()
cuckoo = Dataset("cuckoo", full=false)
samples = Cuckoo.load_samples(cuckoo, inds=1:3)


extractor = cuckoo.extractor


samples
unique(vec(cuckoo.family))

