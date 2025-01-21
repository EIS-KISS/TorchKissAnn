//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

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
