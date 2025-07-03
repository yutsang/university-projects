VERSION 1.0 CLASS
BEGIN
  MultiUse = -1  'True
END
Attribute VB_Name = "Product"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = True
Option Explicit

' Example Product class structure
Private pName As String
Private pPrice As Double
Private pStock As Long

Public Property Get Name() As String
    Name = pName
End Property
Public Property Let Name(val As String)
    pName = val
End Property

Public Property Get Price() As Double
    Price = pPrice
End Property
Public Property Let Price(val As Double)
    pPrice = val
End Property

Public Property Get Stock() As Long
    Stock = pStock
End Property
Public Property Let Stock(val As Long)
    pStock = val
End Property
