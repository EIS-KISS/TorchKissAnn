#include "print.h"

#include "log.h"

void printDataVect(const std::vector<eis::DataPoint>& in)
{
	std::cout<<"omega, real, imag\n";
	for(const eis::DataPoint& res : in)
			std::cout<<res.omega<<','<<res.im.real()<<','<<res.im.imag()<<'\n';
}

void printSvmNode(const svm_node* nodeArray)
{
	for(size_t i = 0; nodeArray[i].index >= 0 ; ++i)
	{
		Log(Log::DEBUG, false)<<nodeArray[i].index<<":"<<nodeArray[i].value<<" ";
	}
	Log(Log::DEBUG)<<"";
}

void printSvmProblem(svm_problem problem)
{
	for(int i = 0; i < problem.l; ++i)
	{
		Log(Log::DEBUG)<<"line "<<i;
		Log(Log::DEBUG)<<problem.y[i]<<"---";
		for(int k = 0; problem.x[i][k].index >= 0; ++k)
		{
			int index = problem.x[i][k].index;
			double value = problem.x[i][k].value;
			Log(Log::DEBUG, false)<<index<<":"<<value<<" ";
		}
		Log(Log::DEBUG)<<"";
	}
}

void svmPrintFunction(const char* str)
{
	Log(Log::DEBUG, false)<<str;
}
