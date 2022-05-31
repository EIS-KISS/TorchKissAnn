#pragma once
#include <libsvm/svm.h>
#include "svmloader.h"
#include "../data/generators/trashgenerator.h"

namespace svm
{
    torch::Tensor use(torch::Tensor input, std::string filenName);

    bool filter(torch::Tensor input, std::string filenName);

    void cross();

    void train(bool oneClass = false);

        template <typename DatasetSelf>
    double test(svm_model* model,
                torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset,
                bool oneClass = false)
    {
        SvmLoader<DatasetSelf> svmloader(dataset);
        struct svm_problem problem = svmloader.getSvmProblem(SIZE_MAX, oneClass);
        int correct = 0;
        double acc;
        for(int i = 0; i < problem.l; ++i)
        {
            double prediction = svm_predict(model, problem.x[i]);
            if(prediction == problem.y[i] || (oneClass && prediction < 0))
                correct++;
            else
                Log(Log::DEBUG)<<"should be "<<(oneClass ? -1.0 : problem.y[i])<<" is "<<svm_predict(model, problem.x[i]);
        }

        acc = static_cast<double>(correct)/problem.l;

        if(oneClass)
        {
            int trashCorrect = 0;
            TrashDataset trashData;
            SvmLoader<TrashDataset> trashSvmloader(&trashData);
            struct svm_problem trashProblem = trashSvmloader.getSvmProblem(1000, oneClass);
            for(int i = 0; i < trashProblem.l; ++i)
            {
                double prediction = svm_predict(model, trashProblem.x[i]);
                if(prediction > 0)
                    trashCorrect++;
                else
                {
                    Log(Log::DEBUG)<<"trashSvmloader should be "<<1.0<<" is "<<svm_predict(model, problem.x[i]);
                }
            }
            double trashAcc = static_cast<double>(trashCorrect)/trashProblem.l;

            Log(Log::INFO)<<"Svm trash accuracy: "<<trashAcc;
            Log(Log::INFO)<<"Svm train accuracy: "<<acc;
            acc = (acc + trashAcc)/2;
        }

        Log(Log::INFO)<<"Svm accuracy: "<<acc;
        SvmLoader<DatasetSelf>::distroySvmProblem(problem);
        return acc;
    }
}
