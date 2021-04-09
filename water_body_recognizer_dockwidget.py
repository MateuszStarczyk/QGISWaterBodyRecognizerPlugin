# -*- coding: utf-8 -*-
"""
/***************************************************************************
 WaterBodyRecognizerDockWidget
                                 A QGIS plugin
 The plugin allows to recognize water bodies on satellite image.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2021-03-03
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Mateusz Starczyk
        email                : starczyk.mateusz@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

from PyQt5.QtWidgets import QDockWidget
from PyQt5.uic import loadUiType
from PyQt5.QtCore import pyqtSignal

from qgis.core import QgsMapLayerProxyModel

FORM_CLASS, _ = loadUiType(os.path.join(os.path.dirname(__file__), 'water_body_recognizer_dockwidget.ui'))


class WaterBodyRecognizerDockWidget(QDockWidget, FORM_CLASS):
    """ The main widget class for interaction with UI
    """
    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        """ Constructor
        """
        super().__init__(parent)
        self.setupUi(self)
        self.mMapLayerComboBox.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.mPolygonLayerComboBox.setFilters(QgsMapLayerProxyModel.PolygonLayer)

    def closeEvent(self, event):
        """ When plugin is closed
        """
        self.closingPlugin.emit()
        event.accept()
