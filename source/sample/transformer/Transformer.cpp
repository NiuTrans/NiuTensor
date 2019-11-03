/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University. 
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 */

#include <math.h>
#include <time.h>
#include "Transformer.h"
#include "T2TModel.h"
#include "T2TUtility.h"
#include "T2TTrainer.h"
#include "T2TPredictor.h"
#include "T2TTester.h"
#include "../../tensor/XDevice.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/XGlobal.h"

namespace transformer
{

int TransformerMain(int argc, const char ** argv)
{
    if(argc == 0)
        return 1;

    char ** args = new char*[argc];
    for(int i = 0; i < argc; i++){
        args[i] = new char[strlen(argv[i]) + 1];
        strcpy(args[i], argv[i]);
    }

    tmpFILE = fopen("tmp.txt", "wb");

    ShowParams(argc, args);

    bool isBeamSearch = false;
    char * trainFN = new char[MAX_LINE_LENGTH];
    char * modelFN = new char[MAX_LINE_LENGTH];
    char * testFN = new char[MAX_LINE_LENGTH];
    char * outputFN = new char[MAX_LINE_LENGTH];

    LoadParamString(argc, args, "train", trainFN, "");
    LoadParamString(argc, args, "model", modelFN, "");
    LoadParamString(argc, args, "test", testFN, "");
    LoadParamString(argc, args, "output", outputFN, "");
    LoadParamBool(argc, args, "beamsearch", &isBeamSearch, false);

    srand((unsigned int)time(NULL));

    T2TTrainer trainer;
    trainer.Init(argc, args);

    T2TModel model;
    model.InitModel(argc, args);
    
    /* learn model parameters */
    if(strcmp(trainFN, ""))
        trainer.Train(trainFN, testFN, strcmp(modelFN, "") ? modelFN : "checkpoint.model", &model);
    
    /* save the final model */
    if(strcmp(modelFN, "") && strcmp(trainFN, ""))
        model.Dump(modelFN);
    
    /* load the model if neccessary */
    if(strcmp(modelFN, ""))
        model.Read(modelFN);

    /* test the model on the new data */
    if(strcmp(testFN, "") && strcmp(outputFN, "")){
        /* beam search */
        if(isBeamSearch){
            T2TTester searcher;
            searcher.Init(argc, args);
            searcher.Test(testFN, outputFN, &model);
        }

        /* forced decoding */
        else{
            T2TTrainer tester;
            tester.Init(argc, args);
            tester.Test(testFN, outputFN, &model);
        }
    }

    delete[] trainFN;
    delete[] modelFN;
    delete[] testFN;
    delete[] outputFN;

    for(int i = 0; i < argc; i++)
        delete[] args[i];
    delete[] args;

    fclose(tmpFILE);

    return 0;
}

}
