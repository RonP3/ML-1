
def sfs(X, y, X_test, y_test, clf, max_features=None):
    if not max_features:
        max_features = X.shape[1] - 1
    features = set(X.keys())
    selected = set([])
    global_max_acc = 0
    while len(selected) < max_features:
        max_acc = 0
        argmax = None
        for feature in features:
            test_features = selected.copy().union({feature})
            clf.fit(X[test_features], y)
            acc = clf.score(X_test[test_features], y_test)
            if acc > max_acc:
                argmax = feature
        if argmax:
            if global_max_acc < max_acc:
                break
            selected.add(argmax)
            features.remove(argmax)
            global_max_acc = max_acc
    return selected


def sbs(X, y, X_test, y_test, clf, min_features=None):
    if not min_features:
        min_features = 0
    features = set(X.keys())
    global_max_acc = 0
    while len(features) > min_features:
        max_acc = 0
        argmax = None
        for feature in features:
            test_features = features - {feature}
            clf.fit(X[test_features], y)
            acc = clf.score(X_test[test_features], y_test)
            if acc > max_acc:
                argmax = feature
        if argmax:
            if global_max_acc < max_acc:
                break
            features.remove(argmax)
            global_max_acc = max_acc
    return features


def bds(X, y, X_test, y_test, clf):
    features = set(X.keys())
    selected = set([])
    removed = set([])
    global_max_acc = 0
    while len(selected) != len(features) - len(removed):
        max_acc = 0
        argmax = None
        for feature in features - selected - removed:
            test_features = selected.copy().union({feature})
            clf.fit(X[test_features], y)
            acc = clf.score(X_test[test_features], y_test)
            if acc > max_acc:
                argmax = feature
        if argmax:
            if global_max_acc < max_acc:
                break
            selected.add(argmax)
            global_max_acc = max_acc

        if len(selected) == len(features) - len(removed):
            break

        argmax = None
        for feature in features - selected - removed:
            test_features = features - removed - {feature}
            clf.fit(X[test_features], y)
            acc = clf.score(X_test[test_features], y_test)
            if acc > max_acc:
                argmax = feature
        if argmax:
            if global_max_acc < max_acc:
                break
            removed.add(argmax)
            global_max_acc = max_acc

    return selected
