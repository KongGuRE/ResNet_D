import datetime

end = datetime.datetime.now()
start = datetime.datetime.now()
total_result: datetime.timedelta = end-end

for i in range(200000):
    print(i, end='')

end = datetime.datetime.now()

result = end - start

print(result)

total_result = total_result + result

print(total_result)
