#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Extraction launcher interface
@author: Alexandre Laurent
'''

import sys
import os

from PyQt4.QtGui import *
from PyQt4.QtCore import *

#from ginnipi import mainUI, about, configuration_form
from nighres.lesion_tool.interface import Ui_LesionTool
from nighres.lesion_tool.lesion_pipeline import Lesion_extractor
import subprocess

#import ginnipi
#from ginnipi.toolbox.xnat import CurlXnat
#from ginnipi.toolbox.utilities import tstamp, mkdir_p
#from ginnipi.pipeline import Pipeline, AbaciConfiguration

class LesionTool(QMainWindow, Ui_LesionTool):
    '''
    GUI for the pipeline launcher
    '''
    def __init__(self, parent=None):
        '''
        Creates the main UI and tweaks table names,
        populates pop-up menus...
        '''
        # setup and wire-up mainUI
        super(LesionTool, self).__init__(parent)
        self.setupUi(self)
        self.connectActions()
        
        # set current directory as working directory by default
        #self.workDir.setText(QString(os.path.abspath('.')))
        
   
    def connectActions(self):
        '''
        Wiring the using interface
        Connecting the buttons and widgets with various methods
        '''
        # buttons,checkboxes
        #self.connectButton.clicked.connect(self.xnatConnect)
        #self.disconnectButton.clicked.connect(self.xnatDisconnect)
        # use lambda function when specifying arguments:
        #self.projectsListWidget.itemClicked.connect(lambda: self.fetchExperiments(str(self.projectsListWidget.currentItem().text())))        #QObject.connect(self.projectsListWidget,SIGNAL("clicked(QModelIndex)",lambda: self.fetchExperiments(self.projectsListModel)))
        #self.selectAll.clicked.connect(self.selectAllExperiments)
        #self.unselectAll.clicked.connect(self.unselectAllExperiments)
        #self.selectionToggle.clicked.connect(self.toggleExperiments)
        #self.destroyed.connect(self.xnatDisconnect)
        #self.pipelinesSelector.activated.connect(self.setupPipeline)
        self.runButton.clicked.connect(self.extract)
         
         
        # Menu bar
        #self.actionConfigure.triggered.connect(self.editConfiguration)
        #self.actionAbout.triggered.connect(self.displayAbout)
        #self.actionQuit.triggered.connect(self.xnatDisconnect)
        self.actionQuit.triggered.connect(app.exit)
        
        # T1 selector
        mainSelector = QFileDialog(parent=self,
                                    caption=QString('Pick a T1 image ...'),
                                    directory=QString('/')
                                )
        mainSelector.setFileMode(QFileDialog.AnyFile)
        self.connect(mainSelector,SIGNAL('fileSelected(QString)'), self.main,SLOT('setText(QString)'))
        self.pushButton_main.clicked.connect(mainSelector.exec_)
        
        # FLAIR selector
        accSelector = QFileDialog(parent=self,
                                    caption=QString('Pick a FLAIR image ...'),
                                    directory=QString('/')
                                )
        accSelector.setFileMode(QFileDialog.AnyFile)
        self.connect(accSelector,SIGNAL('fileSelected(QString)'), self.acc,SLOT('setText(QString)'))
        self.pushButton_acc.clicked.connect(accSelector.exec_)
        
        # InputDir selector
        inputdirSelector = QFileDialog(parent=self,
                                      caption=QString('Select the directory where belong the subjects ...'),
                                      directory=QString(os.getenv('HOME')))
        inputdirSelector.setFileMode(QFileDialog.DirectoryOnly)
        self.connect(inputdirSelector,SIGNAL('fileSelected(QString)'), self.inputDir,SLOT('setText(QString)'))
        self.pushButton_inputDir.clicked.connect(inputdirSelector.exec_)
        
        # SUBJECTS text selector
        subfileSelector = QFileDialog(parent=self,
                                    caption=QString('Pick a list of subjects (.txt) ...'),
                                    directory=QString('/')
                                )
        subfileSelector.setFileMode(QFileDialog.AnyFile)
        self.connect(subfileSelector,SIGNAL('fileSelected(QString)'), self.subfile,SLOT('setText(QString)'))
        self.pushButton_subfile.clicked.connect(subfileSelector.exec_)
        
        # Atlas selector
        atlasSelector = QFileDialog(parent=self,
                                    caption=QString('Pick the atlas file for brain segmentation ...'),
                                    directory=QString('/')
                                )
        atlasSelector.setFileMode(QFileDialog.AnyFile)
        self.connect(atlasSelector,SIGNAL('fileSelected(QString)'), self.atlas,SLOT('setText(QString)'))
        self.pushButton_atlas.clicked.connect(atlasSelector.exec_)
        
        # Workdir selector
        workdirSelector = QFileDialog(parent=self,
                                      caption=QString('Select the working directory...'),
                                      directory=QString(os.getenv('HOME')))
        workdirSelector.setFileMode(QFileDialog.DirectoryOnly)
        self.connect(workdirSelector,SIGNAL('fileSelected(QString)'), self.workDir,SLOT('setText(QString)'))
        self.pushButton_workDir.clicked.connect(workdirSelector.exec_)
    
     
             
    def extract(self):
        '''
        1. Download data, run it
        2. Generate a workflow object with the supplied parameters
        3. Run the pipeline (actually spawns a python script), possibly for days
        4. Upload the results (optionally)
        Needless to say this would benefit from being handled with a Qt Thread
        '''
        
        wf_name = str(self.name.text())
        base_dir = str(self.workDir.text())
        input_dir = str(self.inputDir.text())
        subjects = str(self.subfile.text())
        main = str(self.main.text())
        acc = str(self.acc.text())
        atlas = str(self.atlas.text())
        
        try:
            subprocess.call(['exec.py',wf_name,base_dir,input_dir,subjects,main,acc,atlas]) 
            #wf.run('SLURM',plugin_args={'sbatch_args': '-p gindev'})
        except:
            print("Erreur système, autodestruction dans 3 , 2 , 1 ...")
               




if __name__=='__main__':
    app = QApplication(sys.argv)
    
    obj = LesionTool()
    obj.show()
    
    app.exec_()