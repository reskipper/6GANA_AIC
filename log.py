#!/usr/bin/env python
# coding=utf-8

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import os
import logging
import logging.handlers

# DEBUG,INFO,WARNING,ERROR
level = logging.INFO


def getLogger():
    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(level)
        tracepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'log', 'competition.log'))
        os.makedirs(os.path.dirname(tracepath), exist_ok=True)
        # 设置按文件大小分隔文件,1个文件10M
        file_handler = logging.handlers.RotatingFileHandler(tracepath, maxBytes=10485760, backupCount=100)
        # 设置文件格式
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
        logger.addHandler(file_handler)
    return logger


logger = getLogger()
