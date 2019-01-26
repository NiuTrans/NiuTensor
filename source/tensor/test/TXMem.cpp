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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-6-24
 */

#include "../XGlobal.h"
#include "../XUtility.h"
#include "TXMem.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* case 1: test memory pool class */
bool TestXMemCase1()
{
    bool ok = true;
    int caseNum = 1000;
    int blcokSize = 16;
    int testNum = caseNum * 10;

    int devIDs[2];
    int devNum = 1;
    devIDs[0] = -1;

    /*if (GDevs.nGPU > 0) {
        devIDs[1] = 0;
        devNum = 2;
        devIDs[0] = 0;
        devNum = 1;
    }*/

    int * buf = new int[blcokSize * 10];

    for (int id = 0; id < devNum; id++) {
        int devID = devIDs[id];
        for (int iter = 0, scalar = 1; iter < 3; iter++) {
            XMem mem;
            mem.Initialize(devID, FREE_ON_THE_FLY, blcokSize * sizeof(int) * scalar * scalar, 1000, 0);
            mem.SetIndex(10000, blcokSize * sizeof(int) / 2);

            srand(907);

            int ** p = new int*[caseNum];
            int * size = new int[caseNum];

            for (int i = 0; i < caseNum; i++) {
                p[i] = NULL;
                size[i] = rand() % (2 * blcokSize);
            }

            for (int i = 0; i < testNum * scalar; i++) {
                testxmemid++;
                int j = rand() % caseNum;

                //fprintf(stderr, "%d %d %d\n", testxmemid, j, ok);
                //fprintf(stderr, "iter %d %d %d\n", iter, i, j);

                if (p[j] == NULL) {
                    p[j] = (int*)mem.AllocStandard(mem.devID, size[j] * sizeof(int));
                    for (int k = 0; k < size[j]; k++)
                        buf[k] = j;
                    XMemCopy(p[j], devID, buf, -1, sizeof(int) * size[j]);
                }
                else {
                    mem.ReleaseStandard(mem.devID, p[j], size[j] * sizeof(int));
                    for (int k = 0; k < size[j]; k++)
                        buf[k] = -1;
                    XMemCopy(p[j], devID, buf, -1, sizeof(int) * size[j]);
                    p[j] = NULL;
                }

                for (int k = 0; k < caseNum; k++) {
                    if (p[k] != NULL) {
                        XMemCopy(buf, -1, p[k], devID, sizeof(int) * size[k]);
                        for (int o = 0; o < size[k]; o++) {
                            if (buf[o] != k) {
                                ok = false;
                            }
                        }
                    }
                }
                
                /*MPieceNode * entry = NULL;
                MPieceNode * node = NULL;
                
                entry = mem.memIndex + mem.indexEntryNum + mem.FindIndexEntry(112);
                
                int cc = 0;
                node = entry->next;
                while(node != NULL){
                    fprintf(stderr, "%d ", cc++);
                    if(node->size == 0){
                        MPieceNode * next = node->next;
                        node = next;
                    }
                    else{
                        CheckNTErrors(node->pReal != NULL, "Illegal pointer!");
                        node = node->next;
                    }
                }
                fprintf(stderr, "\n");*/
                
                /*int ccc = 0;
                bool hhh = recordp != NULL ? false : true;
                for(int i = 0; i < mem.indexEntryNum; i++){
                    MPieceNode * entry = mem.memIndex + mem.indexEntryNum + i;
                    
                    MPieceNode * last = entry;
                    MPieceNode * node = entry->next;
                    
                    ccc = 0;
                    while(node != NULL){
                        CheckNTErrors(node->pre == last, "XSomething is wrong!");
                        CheckNTErrors(last->next == node, "XSomething is wrong!");
                        
                        last = node;
                        
                        ccc++;
                        if(node->pReal == recordp){
                            hhh = true;
                        }
                        
                        if(node->size == 0){
                            MPieceNode * next = node->next;
                            node = next;
                        }
                        else{
                            CheckNTErrors(node->pReal != NULL, "Illegal pointer!");
                            node = node->next;
                        }
                    }
                }
                
                if(!hhh){
                    int nnn = 0;
                }*/
            }

            delete[] p;
            delete[] size;
            scalar *= 2;
        }
    }

    delete[] buf;

    return ok;
}

/* test for memory pool class */
bool TestXMem()
{
    XPRINT(0, stdout, "[Test] Memory pool ... Began\n");
    bool returnFlag = true;
    bool caseFlag = true;

    double startT = GetClock();

    /* case 1 test */
    caseFlag = TestXMemCase1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    if (returnFlag) {
        XPRINT(0, stdout, ">> All Passed!\n");
    }
    else
        XPRINT(0, stdout, ">> Failed!\n");

    double endT = GetClock();

    XPRINT1(0, stdout, "[Test] Finished (took %.3lfms)\n\n", endT - startT);

    return returnFlag;
}

} // namespace nts(NiuTrans.Tensor)
