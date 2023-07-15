# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:50:32 2023

@author: starw
"""

import pandas as pd
from molmass import Formula, ELEMENTS


def getData():
    
    #remove un-needed columns
    df = pd.read_excel('data.xlsx')
    df = df.drop('Mineral Name (plain)', axis=1)
    df = df.drop('Oldest Known Age (Ma)', axis=1)
    df = df.drop('Crystal Systems', axis=1)
    df = df.drop('Unnamed: 5', axis=1)
    
    #drop all rows that are more than 4 elements or contain bad charactes
    df = df[df['Chemistry Elements'].apply(lambda x: len(x.split()) <= 4)]
    df = df[df['IMA Chemistry (plain)'].apply(lambda x: ',' not in x and '·' not in x and 'x' not in x and '0' not in x and '?' not in x and '£' not in x and '-' not in x)]
    
    #determine the composition of each compound
    df['percentages'] = df.apply(splitFormula, axis=1)
    
    #split results into multiple rows
    df['E1Percent'] = df['percentages'].apply(lambda x: x[0])
    df['E2Percent'] = df['percentages'].apply(lambda x: x[1])
    df['E3Percent'] = df['percentages'].apply(lambda x: x[2])
    df['E4Percent'] = df['percentages'].apply(lambda x: x[3])
    df['E1Symbol'] = df['percentages'].apply(lambda x: x[4])
    df['E2Symbol'] = df['percentages'].apply(lambda x: x[5])
    df['E3Symbol'] = df['percentages'].apply(lambda x: x[6])
    df['E4Symbol'] = df['percentages'].apply(lambda x: x[7])
    
    #Take each element and break it down into numerical characteristics
    df = breakdownElements(df)
    
    return df
    
'''
The following function will take the chemical formula and calculate the 
mass composition percentage for each formula.
'''
def splitFormula(row):
    formula = row['IMA Chemistry (plain)']
    
    fractions = []
    symbols = []
    
    # If the formula has only one element the molmass composition funciton
    # will not read it in, so skip that function call and list the percentage as 100%
    if len(row['Chemistry Elements'].split()) == 1:
        e = ELEMENTS[formula]
        symbols.append(e.symbol)
        fractions.append(1)
        
    else:
        # use the molmass library to read in the formula and obtain the percentages
        f = Formula(formula)
        formulaComposition = f.composition().dataframe()
        for i in range(len(formulaComposition)):
            percentage = formulaComposition['Fraction'][i]
            symbol = formulaComposition.index[i]
            fractions.append(percentage)
            symbols.append(symbol)
    
    #Pad arrays with 0's or spaces to be uniform.
    if len(fractions) < 4:
        for i in range(len(fractions), 4):
            fractions.append(0)
            
    if len(symbols) < 4:
        for i in range(len(symbols), 4):
            symbols.append('')

    return fractions + symbols
        
    
def breakdownElements(df):
    # Get the atomic attributes for each element
    for i in range(1,5):
        df[f'E{i}Chars'] = df.apply(lambda x: getElementCharacteristics(x, i), axis=1)

    return removeColumns(df)
    
# The following function will use the molmass library to create the 14 item
# feature vector for the given element
def getElementCharacteristics(row, elementNum):
    symbol = row[f'E{elementNum}Symbol']
    percentage = row[f'E{elementNum}Percent']
    info = [percentage]
    if symbol != '':
        e = ELEMENTS[symbol]
        
        info.append(e.number)
        info.append(e.period)
        info.append(e.group)
        info.append(e.series)
        info.append(e.mass)
        info.append(e.eleneg)
        info.append(e.eleaffin)
        info.append(e.covrad)
        info.append(e.atmrad)
        info.append(e.vdwrad)
        info.append(e.tboil)
        info.append(e.tmelt)
        info.append(e.density)
    else:
        info = [0 for i in range(14)]
    return info

# Remove the unused columns
def removeColumns(df):
    df = df.drop('IMA Chemistry (plain)', axis=1)
    df = df.drop('Chemistry Elements', axis=1)
    df = df.drop('percentages', axis=1)
    df = df.drop('E1Symbol', axis=1)
    df = df.drop('E2Symbol', axis=1)
    df = df.drop('E3Symbol', axis=1)
    df = df.drop('E4Symbol', axis=1)
    
    return df
        