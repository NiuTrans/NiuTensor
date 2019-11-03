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
 *
 * We define various names here
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-05
 * It was really HOT these days. I can't imagine it is SO hot here in Shenyang!
*/

#ifndef __XNAME_H__
#define __XNAME_H__

namespace nts { // namespace nts(NiuTrans.Tensor)

/* math operations */
#define MATH_BASE               0x00001000

#define MATH_ABSOLUTE           MATH_BASE + 1
#define MATH_CEIL               MATH_ABSOLUTE + 1
#define MATH_EXP                MATH_CEIL + 1
#define MATH_FLOOR              MATH_EXP + 1
#define MATH_ISNONZERO          MATH_FLOOR + 1
#define MATH_ISZERO             MATH_ISNONZERO + 1
#define MATH_LOG                MATH_ISZERO + 1
#define MATH_SQRT               MATH_LOG + 1
#define MATH_SQUARE             MATH_SQRT + 1
#define MATH_SIN                MATH_SQUARE + 1
#define MATH_COS                MATH_SIN + 1
#define MATH_TAN                MATH_COS + 1
#define MATH_ROUND              MATH_TAN + 1

#define MATH_CLIP               MATH_ROUND + 1
#define MATH_DIV                MATH_CLIP + 1
#define MATH_DIVDIM             MATH_DIV + 1
#define MATH_MASK               MATH_DIVDIM + 1
#define MATH_MATRIXMUL          MATH_MASK + 1
#define MATH_MATRIXMULBATCHED   MATH_MATRIXMUL + 1
#define MATH_MAX                MATH_MATRIXMULBATCHED + 1
#define MATH_MIN                MATH_MAX + 1
#define MATH_MULTIPLY           MATH_MIN + 1
#define MATH_MULTIPLYDIM        MATH_MULTIPLY + 1
#define MATH_MULTIPLYBROADCAST  MATH_MULTIPLYDIM + 1
#define MATH_NEGATE             MATH_MULTIPLYBROADCAST + 1
#define MATH_NORMALIZE          MATH_NEGATE + 1
#define MATH_POWER              MATH_NORMALIZE + 1
#define MATH_SCALEANDSHIFT      MATH_POWER + 1
#define MATH_MULANDSHIFT        MATH_SCALEANDSHIFT + 1
#define MATH_SCALE              MATH_MULANDSHIFT + 1
#define MATH_DESCALE            MATH_SCALE + 1
#define MATH_SHIFT              MATH_DESCALE + 1
#define MATH_MOD                MATH_SHIFT + 1
#define MATH_SIGN               MATH_MOD + 1
#define MATH_SUB                MATH_SIGN + 1
#define MATH_SUBDIM             MATH_SUB + 1
#define MATH_SUM                MATH_SUBDIM + 1
#define MATH_SUMDIM             MATH_SUM + 1
#define MATH_SUMBROADCAST       MATH_SUMDIM + 1

#define REDUCE                  MATH_SUMBROADCAST + 1
#define REDUCE_REDUCEMAX        REDUCE + 1
#define REDUCE_REDUCEMEAN       REDUCE_REDUCEMAX + 1
#define REDUCE_REDUCESUM        REDUCE_REDUCEMEAN + 1
#define REDUCE_REDUCESUMSQUARED REDUCE_REDUCESUM + 1
#define REDUCE_REDUCEVARIANCE   REDUCE_REDUCESUMSQUARED + 1

/* data and shape related operations */
#define DATA_BASE               MATH_BASE * 2
#define GETANDSET               DATA_BASE + 1
#define GETANDSET_CONVERTDATATYPE GETANDSET + 1
#define GETANDSET_SELECT        GETANDSET_CONVERTDATATYPE + 1

#define MOVEMENT                GETANDSET_SELECT + 1
#define MOVEMENT_COPYINDEXED    MOVEMENT + 1
#define MOVEMENT_COPYVALUES     MOVEMENT_COPYINDEXED + 1
#define MOVEMENT_GATHER         MOVEMENT_COPYVALUES + 1
#define MOVEMENT_DROPOUTWITHINDEX         MOVEMENT_GATHER + 1

#define SHAPE                   MOVEMENT_DROPOUTWITHINDEX + 1
#define SHAPE_CONCATENATE       SHAPE + 1
#define SHAPE_MERGE             SHAPE_CONCATENATE + 1
#define SHAPE_MERGE_LIST        SHAPE_MERGE + 1
#define SHAPE_PERMUTE           SHAPE_MERGE_LIST + 1
#define SHAPE_RESHAPE           SHAPE_PERMUTE + 1
#define SHAPE_SPLIT             SHAPE_RESHAPE + 1
#define SHAPE_SPLIT_LIST        SHAPE_SPLIT + 1
#define SHAPE_STACK             SHAPE_SPLIT_LIST + 1
#define SHAPE_SQUEEZE           SHAPE_STACK + 1
#define SHAPE_TRANSPOSE         SHAPE_SQUEEZE + 1
#define SHAPE_UNSQUEEZE         SHAPE_TRANSPOSE + 1

#define SORT                    SHAPE_UNSQUEEZE + 1
#define SORT_SORT               SORT + 1
#define SORT_TOPK               SORT_SORT + 1

/* activation functions */
#define FUNCTION_BASE           DATA_BASE * 2
#define FUNC_DROPOUT            FUNCTION_BASE + 1
#define FUNC_HARDTANH           FUNC_DROPOUT + 1
#define FUNC_IDENTITY           FUNC_HARDTANH + 1
#define FUNC_LOGSOFTMAX         FUNC_IDENTITY + 1
#define FUNC_RECTIFY            FUNC_LOGSOFTMAX + 1
#define FUNC_SIGMOID            FUNC_RECTIFY + 1
#define FUNC_SOFTMAX            FUNC_SIGMOID + 1

#define LOSS_BASE               FUNCTION_BASE * 2
#define LOSS_CROSSENTROPY       LOSS_BASE + 1

/* get operator name */
const char * GetOPName(int type);

} // namespace nts(NiuTrans.Tensor)

#endif // __XNAME_H__
