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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-05
 */

#include "XName.h"

namespace nts { // namespace nts(NiuTrans.Tensor)
    
/* get operator name */
const char * GetOPName(int type)
{
    if ((type & MATH_BASE) != 0){
        if (type == MATH_ABSOLUTE)
            return "M_ABSOLUTE";
        else if (type == MATH_CEIL)
            return "M_CEIL";
        else if (type == MATH_EXP)
            return "M_EXP";
        else if (type == MATH_FLOOR)
            return "M_FLOOR";
        else if (type == MATH_ISNONZERO)
            return "M_ISNONZERO";
        else if (type == MATH_ISZERO)
            return "M_ISZERO";
        else if (type == MATH_LOG)
            return "M_LOG";
        else if (type == MATH_SQRT)
            return "M_SQRT";
        else if (type == MATH_SQUARE)
            return "M_SQUARE";
        else if (type == MATH_SIN)
            return "M_SIN";
        else if (type == MATH_COS)
            return "M_COS";
        else if (type == MATH_TAN)
            return "M_TAN";
        else if (type == MATH_ROUND)
            return "M_ROUND";
        else if (type == MATH_CLIP)
            return "M_CLIP";
        else if (type == MATH_DIV)
            return "M_DIV";
        else if (type == MATH_DIVDIM)
            return "M_DIVDIM";
        else if (type == MATH_MASK)
            return "M_MASK";
        else if (type == MATH_MATRIXMUL)
            return "M_MATRIXMUL";
        else if (type == MATH_MATRIXMULBATCHED)
            return "M_MATRIXMULBATCHED";
        else if (type == MATH_MULTIPLY)
            return "M_MULTIPLY";
        else if (type == MATH_MULTIPLYDIM)
            return "M_MULTIPLYDIM";
        else if (type == MATH_MULTIPLYBROADCAST)
            return "M_MULTIPLYBROADCAST";
        else if (type == MATH_NEGATE)
            return "M_NEGATE";
        else if (type == MATH_NORMALIZE)
            return "M_NORMALIZE";
        else if (type == MATH_POWER)
            return "M_POWER";
        else if (type == MATH_SCALEANDSHIFT)
            return "M_SCALEANDSHIFT";
        else if (type == MATH_SCALE)
            return "M_SCALE";
        else if (type == MATH_DESCALE)
            return "M_DESCALE";
        else if (type == MATH_SHIFT)
            return "M_SHIFT";
        else if (type == MATH_MULANDSHIFT)
            return "M_OPERATION";
        else if (type == MATH_SIGN)
            return "M_SIGN";
        else if (type == MATH_SUB)
            return "M_SUB";
        else if (type == MATH_SUBDIM)
            return "M_SUBDIM";
        else if (type == MATH_SUM)
            return "M_SUM";
        else if (type == MATH_SUMDIM)
            return "M_SUMDIM";
        else if (type == MATH_SUMBROADCAST)
            return "M_SUMBROADCAST";
        else if (type == REDUCE_REDUCEMAX)
            return "R_REDUCEMAX";
        else if (type == REDUCE_REDUCEMEAN)
            return "R_REDUCEMEAN";
        else if (type == REDUCE_REDUCESUM)
            return "R_REDUCESUM";
        else if (type == REDUCE_REDUCESUMSQUARED)
            return "R_REDUCESUMSQUARED";
        else if (type == REDUCE_REDUCEVARIANCE)
            return "R_REDUCEVARIANCE";
    }
    else if ((type & DATA_BASE) != 0){
        if (type == GETANDSET_SELECT)
            return "G_SELECT";
        else if (type == MOVEMENT_COPYINDEXED)
            return "M_COPYINDEXED";
        else if (type == MOVEMENT_COPYVALUES)
            return "M_COPYVALUES";
        else if (type == MOVEMENT_GATHER)
            return "M_GATHER";
        else if (type == MOVEMENT_DROPOUTWITHINDEX)
            return "M_DROPOUTWITHINDEX";
        else if (type == SHAPE_CONCATENATE)
            return "S_CONCATENATE";
        else if (type == SHAPE_MERGE)
            return "S_MERGE";
        else if (type == SHAPE_MERGE_LIST)
            return "S_MERGE_LIST";
        else if (type == SHAPE_PERMUTE)
            return "S_PERMUTE";
        else if (type == SHAPE_RESHAPE)
            return "S_RESHAPE";
        else if (type == SHAPE_SPLIT)
            return "S_SPLIT";
        else if (type == SHAPE_SPLIT_LIST)
            return "S_SPLIT_LIST";
        else if (type == SHAPE_STACK)
            return "S_SHAPE_STACK";
        else if (type == SHAPE_SQUEEZE)
            return "S_SQUEEZE";
        else if (type == SHAPE_TRANSPOSE)
            return "S_TRANSPOSE";
        else if (type == SHAPE_UNSQUEEZE)
            return "S_UNSQUEEZE";
        else if (type == SORT_SORT)
            return "S_SORT";
        else if (type == SORT_TOPK)
            return "S_TOPK";
    }
    else if ((type & FUNCTION_BASE) != 0){
        if (type == FUNC_DROPOUT)
            return "F_DROPOUT";
        else if (type == FUNC_HARDTANH)
            return "F_HARDTANH";
        else if (type == FUNC_IDENTITY)
            return "F_IDENTITY";
        else if (type == FUNC_LOGSOFTMAX)
            return "F_LOGSOFTMAX";
        else if (type == FUNC_RECTIFY)
            return "F_RECTIFY";
        else if (type == FUNC_SIGMOID)
            return "F_SIGMOID";
        else if (type == FUNC_SOFTMAX)
            return "F_SOFTMAX";
    }
    else if ((type & LOSS_BASE) != 0) {
        if (type == LOSS_CROSSENTROPY)
            return "L_CROSSENTROPY";
    }
    
    return "NULL";
}
    
} // namespace nts(NiuTrans.Tensor)

