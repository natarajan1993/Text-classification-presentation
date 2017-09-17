import time

cur_x = 6 # The algorithm starts at x=6
gamma = 0.01 # step size multiplier
precision = 0.00001
previous_step_size = cur_x

def df(x):
    return 4 * x**3 - 9 * x**2

count = 1
while previous_step_size > precision:
    print("Iteration:",count)
    print("Current x:",cur_x)
    print("Previous step size:",previous_step_size)
    print()
    prev_x = cur_x
    cur_x += -gamma * df(prev_x)
    previous_step_size = abs(cur_x - prev_x)
    count += 1 
    time.sleep(0.25)
print("Completed in:",count,"steps") 
print("The local minimum occurs at %f" % cur_x)
