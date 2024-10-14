#pragma once

#include <kisstype/type.h>
#include <libsvm/svm.h>

void printDataVect(const std::vector<eis::DataPoint>& in);
void printSvmNode(const svm_node* nodeArray);
void printSvmProblem(svm_problem problem);
void svmPrintFunction(const char* str);
