import os
import numpy as np
import pandas as pd

from svm_algorithms import SVMPrimal, SVMDual 


def main():
    current_directory = os.getcwd()

    train_file_path = os.path.join(current_directory, 'bank-note', 'train.csv')
    test_file_path = os.path.join(current_directory, 'bank-note', 'test.csv')

    training_data = pd.read_csv(train_file_path, header=None) 
    testing_data = pd.read_csv(test_file_path, header=None) 

    X_train = training_data.iloc[:, 0:4].to_numpy()
    y_train = training_data.iloc[:, 4].to_numpy()
    y_train = np.where(np.isnan(y_train), -1, y_train)
    y_train = np.where(y_train == 0, -1, y_train)

    X_test = testing_data.iloc[:, 0:4].to_numpy()
    y_test = testing_data.iloc[:, 4].to_numpy()
    y_test = np.where(np.isnan(y_test), -1, y_test)
    y_test = np.where(y_test == 0, -1, y_test)

    C_list = [100/873, 500/873, 700/873]
    
    # two
    n_epoch = 100
    gamma_0 = 0.1
    a = 0.001
    
    # #a
    Sch1 = SVMPrimal()
    print("\n 2 (a)")
    for C in C_list:
        Sch1.fit(X_train, y_train, C, n_epoch, 1, gamma_0, a)
        print(f"\n With C = {C}")
        print(f"Weight = {Sch1.weight} ")
        print(f"Average train error = {Sch1.calculate_error(X_train, y_train)} ")
        print(f"Average test error = {Sch1.calculate_error(X_test, y_test)} ")

    # #b
    Sch2 = SVMPrimal()
    print("\n 2 (b)")
    for C in C_list:
        Sch2.fit(X_train, y_train, C, n_epoch, 2, gamma_0)   
        print(f"\n With C = {C}")
        print(f"Weight = {Sch2.weight}")
        print(f"Average train error = {Sch2.calculate_error(X_train, y_train)}") 
        print(f"Average test error = {Sch2.calculate_error(X_test, y_test)}")

    #three
    # a
    print("\n 3 (a)")
    LinSVM = SVMDual()
    for C in C_list:
        LinSVM.fit(X_train, y_train, C, kernel_type='linear')
        print(f"\n With C = {C}")
        print(f"Weight = {LinSVM.weight}")
        print(f"Bias = {LinSVM.bias}")
        print(f"Average train error = {LinSVM.calculate_error(X_train, y_train)}") 
        print(f"Average test error = {LinSVM.calculate_error(X_test, y_test)}")

    # b
    print("\n 3 (b)")
    gamma_list = [0.1, 0.5, 5]
    GSVM = SVMDual()
    sv_c = []
    for C in C_list:
        for gamma in gamma_list:
            GSVM.fit(X_train, y_train, C, kernel_type='gaussian', gamma=gamma)
            print(f"\n With C = {C} and gamma = {gamma}")
            print(f"Weight = {GSVM.weight}")
            print(f"Bias = {GSVM.bias}")
            sv = GSVM.support_vectors 
            print(f"Support vectors number = {len(sv)}")
            print(f"Average train error = {GSVM.calculate_error(X_train, y_train)}") 
            print(f"Average test error = {GSVM.calculate_error(X_test, y_test)}")
            if C == 500/873:
                sv_c.append(np.round(sv, 5))

        if C == 500/873:
            for i in range(len(gamma_list) - 1):
                overlap_n = 0
                for sv in sv_c[i]:
                    if sv in sv_c[i + 1]:
                        overlap_n += 1
                print(f"Overlap for gamma = {gamma_list[i]} and {gamma_list[i+1]}: {overlap_n}")


if __name__ == "__main__":
    main()