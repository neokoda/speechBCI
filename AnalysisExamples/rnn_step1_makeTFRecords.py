#!/usr/bin/env python
# coding: utf-8

# In[1]:


#formats the competitionData into tfRecords for RNN training, including blockwise feature normalization
baseDir = 'c:/Users/LENOVO/Koding/Semester 8/TA/speechBCI/data'

import os
os.makedirs(baseDir+'/derived/tfRecords', exist_ok=True)


# In[ ]:


from makeTFRecordsFromSession import makeTFRecordsFromCompetitionFiles 
from getSpeechSessionBlocks import getSpeechSessionBlocks
blockLists = getSpeechSessionBlocks()

for sessIdx in range(len(blockLists)):
    sessionName = blockLists[sessIdx][0]
    dataPath = baseDir + '/competitionData'
    tfRecordFolder = baseDir + '/derived/tfRecords/'+sessionName
    makeTFRecordsFromCompetitionFiles(sessionName, dataPath, tfRecordFolder)

