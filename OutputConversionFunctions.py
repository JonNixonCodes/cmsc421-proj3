

def convertOutputIndexToChangeInVelocity(chosenIndex):

    if (chosenIndex == 0):
        return (-1,-1)
    if (chosenIndex == 1):
        return (-1,0)
    if (chosenIndex == 2):
        return (-1, 1)
    if (chosenIndex == 3):
        return (0, -1)
    if (chosenIndex == 4):
        return (0, 0)
    if (chosenIndex == 5):
        return (0, 1)
    if (chosenIndex == 6):
        return (1, -1)
    if (chosenIndex == 7):
        return (1, 0)
    if (chosenIndex == 8):
        return (1, 1)


def convertChangeInVelocityToIndex(changeInVelocity):

    (dx, dy) = changeInVelocity

    if dx == -1 and dy == -1:
        return 0
    if dx == -1 and dy == 0:
        return 1
    if dx == -1 and dy == 1:
        return 2
    if dx == 0 and dy == -1:
        return 3
    if dx == 0 and dy == 0:
        return 4
    if dx == 0 and dy == 1:
        return 5
    if dx == 1 and dy == -1:
        return 6
    if dx == 1 and dy == 0:
        return 7
    if dx == 1 and dy == 1:
        return 8


def convertOldAndNewVelocityToChangeInVelocity(oldVelocity, newVelocity):

    oldU, oldV = oldVelocity
    newU, newV = newVelocity

    uDifference = newU - oldU
    vDifference = newV - oldV

    return (uDifference, vDifference)


def convertOldVelocityAndChangeInVelocityToNewVelocity(oldVelocity, changeInVelocity):

    oldU, oldV = oldVelocity
    dU, dV = changeInVelocity

    newU = oldU + dU
    newV = oldV + dV

    return (newU, newV)


def convertChangeInVelocityIndexToVector(velocityIndex):
    listofzeros = [0] * 9
    listofzeros[velocityIndex] = 1
    return listofzeros


def convertChangeInvelocityToVector(changeInVelocity):
    changeInVelocityIndex = convertChangeInVelocityToIndex(changeInVelocity)
    changeInVelocityVector = convertChangeInVelocityIndexToVector(changeInVelocityIndex)
    return changeInVelocityVector


def convertOutputVectorIntoChosenChangeInVelocity(outputWeights):

    bestIndex = 0
    bestIndexValue = 0

    curIndex = 0
    for x in outputWeights:
        if outputWeights[x] > bestIndexValue:
            bestIndexValue = outputWeights[x]
            bestIndex = curIndex
        curIndex += 1

    chosenChangeInVelocity = convertOutputIndexToChangeInVelocity(bestIndex)
    return chosenChangeInVelocity
