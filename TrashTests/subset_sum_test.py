import time
# numbers = [(221, 240253), (223, 356802), (225, 355606), (227, 382438), (229, 386736), (231, 437603), (233, 424687), (234, 550962), (235, 626347), (237, 527176), (238, 566023), (239, 455872), (240, 323213), (241, 328596), (243, 276037), (244, 308471), (245, 441148), (246, 352821), (248, 225720), (250, 135154), (252, 389813), (254, 334134)]
# x = 8425612
# [(87, 34926), (89, 40464), (90, 35020), (92, 45547), (93, 49113), (95, 43501), (97, 45231), (99, 32045), (101, 10941), (102, 10057), (104, 15327), (106, 9628), (107, 23790), (108, 23167), (110, 9802), (111, 13889), (113, 16104), (115, 32174), (117, 20619), (118, 22875), (119, 35890), (121, 46358), (123, 51885), (125, 57050), (127, 54455), (129, 75341), (130, 90033), (132, 76621), (133, 94940), (135, 101886), (137, 104447), (138, 127608), (139, 146585), (141, 131397), (143, 135867), (145, 121193), (146, 151601), (148, 131876), (150, 122380), (152, 138676), (154, 139184), (155, 166686), (157, 158878), (159, 144174), (161, 129095), (163, 129760), (164, 175812), (166, 232068), (167, 209760), (169, 341624), (171, 323037), (172, 290581), (174, 198521), (175, 178252), (176, 222569), (177, 221241), (179, 248843), (180, 260101), (181, 298377), (182, 193803), (184, 108643), (185, 123311), (187, 127090), (188, 413298), (189, 147845)], 65, 7712862, []

numbers = [(87, 34926), (89, 40464), (90, 35020), (92, 45547), (93, 49113), (95, 43501), (97, 45231), (99, 32045), (101, 10941), (102, 10057), (104, 15327), (106, 9628), (107, 23790), (108, 23167), (110, 9802), (111, 13889), (113, 16104), (115, 32174), (117, 20619), (118, 22875), (119, 35890), (121, 46358), (123, 51885), (125, 57050), (127, 54455), (129, 75341), (130, 90033), (132, 76621), (133, 94940), (135, 101886), (137, 104447), (138, 127608), (139, 146585), (141, 131397), (143, 135867), (145, 121193), (146, 151601), (148, 131876), (150, 122380), (152, 138676), (154, 139184), (155, 166686), (157, 158878), (159, 144174), (161, 129095), (163, 129760), (164, 175812), (166, 232068), (167, 209760), (169, 341624), (171, 323037), (172, 290581), (174, 198521), (175, 178252), (176, 222569), (177, 221241), (179, 248843), (180, 260101), (181, 298377), (182, 193803), (184, 108643), (185, 123311), (187, 127090), (188, 413298), (189, 147845)]
x = 7712862

# print(len(numbers))
# exit()
i = 1


def _subset_sum(numbers, n, x, indices):
    global i
    # time.sleep(0.1)
    # print(i, n, indices)
    i += 1
    # Base Cases
    if x == 0:
        return True
    if n == 0 and x != 0:
        return False
    # If last element is greater than x, then ignore it
    if numbers[n - 1][1] > x:
        print('greater than')
        return _subset_sum(numbers, n - 1, x, indices)
    # else, check if x can be obtained by any of the following
    # (a) including the last element

    # print('sunul', n)
    indices.insert(0, numbers[n - 1][0])
    found = _subset_sum(numbers, n - 1, x - numbers[n - 1][1], indices)
    if not found:
        indices.pop(0)
    else:
        return True

    # found = _subset_sum(numbers, n - 1, x, indices)
    # if found:
    #     return True
    # (b) excluding the last element

    found = _subset_sum(numbers, n - 1, x, indices)
    if found:
        return True

    # print('sunul', n)
    # indices.insert(0, numbers[n - 1][0])
    # found = _subset_sum(numbers, n - 1, x - numbers[n - 1][1], indices)
    # if not found:
    #     indices.pop(0)
    return found


idxs = []
print(_subset_sum(numbers, len(numbers), x, idxs))
print(x)
print(numbers)
print(idxs)
print(len(idxs))

print(sum([num for _, num in numbers]))