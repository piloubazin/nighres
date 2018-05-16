#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Extraction launcher interface
@author: Alexandre Laurent
'''

import sys
import os
import pickle

from PyQt4.QtGui import *
from PyQt4.QtCore import *

from nighres.lesion_tool.interface import Ui_LesionTool
from nighres.lesion_tool.lesion_pipeline import Lesion_extractor
import subprocess

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
        
        self.grid_selector.addItems(QStringList(['',
                                                 'normal',
                                                 'highmem',
                                                 'gindev']))
        
        self.SubjectWidget.setHorizontalHeaderItem(1,  QTableWidgetItem(""))
        self.SubjectWidget.setHorizontalHeaderItem(1,  QTableWidgetItem("Subject"))
        self.SubjectWidget.setColumnWidth(0, 30)
        self.SubjectWidget.setColumnWidth(1, 100)
        
        # set current directory as working directory by default
        self.workDir.setText(QString(os.path.abspath('.')))
        
   
    def connectActions(self):
        '''
        Wiring the using interface
        Connecting the buttons and widgets with various methods
        '''
        # buttons,checkboxes
        self.selectAll.clicked.connect(self.selectAllExperiments)
        self.unselectAll.clicked.connect(self.unselectAllExperiments)
        self.selectionToggle.clicked.connect(self.toggleExperiments)

        self.runButton.clicked.connect(self.extract)
                
        self.pushButton_loadInputDir.clicked.connect(self.loadSubjects)
        
        self.actionQuit.triggered.connect(app.exit)
        
        # InputDir selector
        inputdirSelector = QFileDialog(parent=self,
                                      caption=QString('Select the directory where belong the subjects ...'),
                                      directory=QString(os.getenv('HOME'))
                                      )
        inputdirSelector.setFileMode(QFileDialog.DirectoryOnly)
        self.connect(inputdirSelector,SIGNAL('fileSelected(QString)'), self.inputDir,SLOT('setText(QString)'))
        self.pushButton_inputDir.clicked.connect(inputdirSelector.exec_)
        
        # Atlas selector
        atlasSelector = QFileDialog(parent=self,
                                    caption=QString('Pick the atlas file for brain segmentation ...'),
                                    directory=QString(os.getenv('HOME'))
                                    )
        atlasSelector.setFileMode(QFileDialog.AnyFile)
        self.connect(atlasSelector,SIGNAL('fileSelected(QString)'), self.atlas,SLOT('setText(QString)'))
        self.pushButton_atlas.clicked.connect(atlasSelector.exec_)
        
        # Workdir selector
        workdirSelector = QFileDialog(parent=self,
                                      caption=QString('Select the working directory...'),
                                      directory=QString(os.getenv('HOME'))
                                      )
        workdirSelector.setFileMode(QFileDialog.DirectoryOnly)
        self.connect(workdirSelector,SIGNAL('fileSelected(QString)'), self.workDir,SLOT('setText(QString)'))
        self.pushButton_workDir.clicked.connect(workdirSelector.exec_)
    
    def loadSubjects(self):
        '''
        Load contents of input folder for BIDS structure.
        This function is more or less a mix between xnatConnect and fetchExperiments.
        '''
        folder = str(self.inputDir.text())
        if os.path.isdir(folder):
            subjects = os.listdir(folder)
            project = folder.split("/")[-1]
            if project == "":
                project = folder.split("/")[-2]
                
            self.SubjectWidget.setColumnCount(2)
            self.SubjectWidget.setRowCount(len(subjects))
            
            r = 0
            for subject in subjects : 
                item = QTableWidgetItem()
                item.setCheckable = True
                item.setCheckState(Qt.Checked)                
                self.SubjectWidget.setItem(r,0, item)
                self.SubjectWidget.setItem(r,1, QTableWidgetItem(subject))
                r += 1
            self.SubjectWidget.show()
            self.statusBar.showMessage('Retrieved '+str(r)+' subjects')
        else:
            self.statusBar.showMessage('Failed to find any subject in this folder')
            pass
    
    def selectAllExperiments(self):
        '''
        Mark all experiments as checked
        '''
        for i in range(0,self.SubjectWidget.rowCount()):
            self.SubjectWidget.item(i,0).setCheckState(Qt.Checked)
        return 
    
    def unselectAllExperiments(self):
        '''
        Mark all experiments as unchecked
        '''
        for i in range(0,self.SubjectWidget.rowCount()):
            self.SubjectWidget.item(i,0).setCheckState(Qt.Unchecked)
        return 
    
    def toggleExperiments(self):
        '''
        Mark all experiments as unchecked if checked and vice versa
        '''
        rows=list(set([i.row() for i in self.SubjectWidget.selectedIndexes()]))
        if all([self.SubjectWidget.item(row,0).checkState()==Qt.Checked for row in rows]):
            for i in rows:
                self.SubjectWidget.item(i,0).setCheckState(Qt.Unchecked)
        else:
            for i in rows:
                self.SubjectWidget.item(i,0).setCheckState(Qt.Checked)    
        return
             
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
        
        idxes = range(self.SubjectWidget.rowCount())
        rows = list(set([idx for idx in idxes if self.SubjectWidget.item(idx,0).checkState()==Qt.Checked] ))
        subjects = list([str(self.SubjectWidget.item(row,1).text()) for row in rows])
        pickle.dump( subjects, open( "subjects.pkl", "wb" ) )
        
        atlas = str(self.atlas.text())
        grid = str(self.grid_selector.currentText())

        try:
            #wf.run('SLURM',plugin_args={'sbatch_args': '-p gindev'})
            subprocess.call(['exec.py',wf_name,base_dir,input_dir,"subjects.pkl",atlas,grid]) 
            print('Extraction ran successfully')
            
        except:
            print("Erreur système, autodestruction dans 3 , 2 , 1 ...")
               


if __name__=='__main__':
    app = QApplication(sys.argv)
    
    obj = LesionTool()
    obj.show()
    
    app.exec_()