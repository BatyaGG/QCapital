def subset_sum(numbers, n, x, indices, negatives=False):
    # Base Cases
    if x == 0:
        return True
    if n == 0 and x != 0:
        return False
    # If last element is greater than x, then ignore it
    if not negatives:
        if numbers[n - 1] > x:
            return subset_sum(numbers, n - 1, x, indices)
    else:
        if numbers[n - 1] <= x:
            return subset_sum(numbers, n - 1, x, indices)
    # else, check if x can be obtained by any of the following
    # (a) including the last element
    found = subset_sum(numbers, n - 1, x, indices)
    if found:
        return True
    # (b) excluding the last element
    indices.insert(0, numbers[n - 1])
    found = subset_sum(numbers, n - 1, x - numbers[n - 1], indices)
    if not found:
        indices.pop(0)
    return found


a = [-1,-4,-2,-6,-18,-2,-7]
b = [1,4,2,6,18,2,7]
c = a
res = []
subset_sum(c, len(c), -3, res, negatives=True)

print(res)
