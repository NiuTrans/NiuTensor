/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#include <cmath>
#include <ctime>

#include "Transformer.h"
#include "train/T2TTrainer.h"
#include "module/T2TUtility.h"
#include "translate/T2TTranslator.h"
#include "../../tensor/XDevice.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XUtility.h"

namespace transformer
{

int TransformerMain(int argc, const char** argv)
{
    if (argc == 0)
        return 1;

    /* load configurations */
    T2TConfig config(argc, argv);

    srand((unsigned int)time(NULL));

    /* train the model */
    if (strcmp(config.trainFN, "") != 0) {
        ENABLE_GRAD;
        T2TModel model;
        model.InitModel(config);
        T2TTrainer trainer;
        trainer.Init(config);
        trainer.Train(config.trainFN, config.validFN, config.modelFN, &model);
    }

    /* translate the test file */
    if (strcmp(config.testFN, "") != 0 && strcmp(config.outputFN, "") != 0) {
        DISABLE_GRAD;
        T2TModel model;
        model.InitModel(config);
        T2TTranslator translator;
        translator.Init(config);
        translator.Translate(config.testFN, config.srcVocabFN, 
                             config.tgtVocabFN, config.outputFN, &model);
    }

    return 0;
}

}