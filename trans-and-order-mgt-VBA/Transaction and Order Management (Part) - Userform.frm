VERSION 5.00
Begin {C62A69F0-16DC-11CE-9E98-00AA00574A4F} OrderForm 
   Caption         =   "CityU Shop Order Form"
   ClientHeight    =   6909
   ClientLeft      =   100
   ClientTop       =   420
   ClientWidth     =   9800.001
   OleObjectBlob   =   "Transaction and Order Management (Part) - Userform.frx":0000
   StartUpPosition =   1  'CenterOwner
End
Attribute VB_Name = "OrderForm"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Option Explicit

' === Module-level variables ===
Dim dblTotal As Double ' Current order total
Dim dblAccTotal As Double ' Accumulated total for the session

' === Event Handlers ===

Private Sub Label4_Click()
    ' No action required
End Sub

Private Sub lstProducts_Click()
    ' Enable quantity frame when a product is selected
    fmeQuantity.Enabled = True
End Sub

' Update quantity label when scrollbar changes
Private Sub sbrQuantity_Change()
    lblQuantity.Caption = sbrQuantity.Value & " items"
End Sub

' Handle ordering a product
Private Sub btnOrder_Click()
    Dim chrPrice As Double
    Dim inv As Integer
    
    With lstProducts
        inv = CLng(.List(.ListIndex, 2))
        ' If in stock
        If inv > 0 Then
            fmeQuantity.Enabled = True
            ' Add product to order list
            lstOrdered.AddItem .List(.ListIndex, 0)
            chrPrice = CDbl(.List(.ListIndex, 1))
            sbrQuantity.Max = inv
            lstOrdered.List(lstOrdered.ListCount - 1, 1) = sbrQuantity.Value
            lstOrdered.List(lstOrdered.ListCount - 1, 2) = VBA.Format(chrPrice * sbrQuantity.Value, "0.0")
            dblTotal = dblTotal + chrPrice * sbrQuantity.Value
            lblTotal.Caption = VBA.Format(dblTotal, "$#,##0.0")
            .List(.ListIndex, 2) = inv - sbrQuantity.Value
            sbrQuantity.Max = inv - sbrQuantity.Value
            .ListIndex = -1
            sbrQuantity.Value = 1
            fmeQuantity.Enabled = False
        Else
            MsgBox "Selected product is out of stock. Please select another product.", vbExclamation, "CityU Shop"
            fmeQuantity.Enabled = False
            fmeQuantity.Visible = True
        End If
    End With
End Sub

' Enable/disable Remove button based on selection
Private Sub lstOrdered_Change()
    Dim intIndex As Integer, intSelect As Integer
    btnRemove.Enabled = False
    For intIndex = 0 To lstOrdered.ListCount - 1
        If lstOrdered.Selected(intIndex) Then
            btnRemove.Enabled = True
            Exit For
        End If
    Next intIndex
End Sub

' Remove selected items from order
Private Sub btnRemove_Click()
    Dim int_index As Integer, int_index1 As Integer, int_found As Integer, inv As Integer
    With lstOrdered
        For int_index = .ListCount - 1 To 0 Step -1
            If .Selected(int_index) Then
                int_found = -1
                For int_index1 = lstProducts.ListCount - 1 To 0 Step -1
                    If lstProducts.List(int_index1, 0) = .List(int_index, 0) Then
                        int_found = int_index1
                        Exit For
                    End If
                Next int_index1
                If int_found <> -1 Then
                    inv = CLng(.List(int_index, 1))
                    lstProducts.List(int_found, 2) = CLng(lstProducts.List(int_found, 2)) + inv
                End If
                dblTotal = dblTotal - CDbl(.List(int_index, 2))
                lblTotal.Caption = VBA.Format(dblTotal, "$#,##0.0")
                .RemoveItem int_index
            End If
        Next int_index
    End With
    btnRemove.Enabled = False
End Sub

' Finalize order and accumulate total
Private Sub btnProcOrder_Click()
    Dim intIndex1 As Integer
    For intIndex1 = 0 To lstOrdered.ListCount - 1
        dblAccTotal = dblAccTotal + CDbl(lstOrdered.List(intIndex1, 2))
    Next intIndex1
    lstOrdered.Clear
    dblTotal = 0
    lblTotal.Caption = VBA.Format(dblTotal, "$#,##0.0")
End Sub

' Show accumulated value
Private Sub chkAccValue_Click()
    If chkAccValue.Value = True Then
        MsgBox "The accumulated order value is: " & VBA.Format(dblAccTotal, "$#,##0.0"), vbInformation, "CityU Shop"
    End If
    chkAccValue.Value = False
End Sub

' Handle quit button
Private Sub btnQuit_Click()
    Dim Response As Integer
    Response = MsgBox("Do you want to quit?", vbYesNo + vbQuestion, "CityU Shop")
    If Response = vbYes Then
        VBA.Unload OrderForm
    End If
End Sub

' Initialize form and product list
Private Sub UserForm_Initialize()
    fmeQuantity.Enabled = False
    fmeQuantity.Visible = True
    sbrQuantity.Max = 100
    sbrQuantity.SmallChange = 1
    sbrQuantity.Value = 1
    dblTotal = 0
    ' Populate product list from Product sheet
    With lstProducts
        .ColumnCount = 3
        .ColumnWidths = "120;60;50"
        .List = Product.Range("a1").CurrentRegion.Value
    End With
End Sub

' Prevent closing the form except via Quit
Private Sub UserForm_QueryClose(Cancel As Integer, CloseMode As Integer)
    If CloseMode <> 1 Then
        Cancel = 1
        VBA.MsgBox "Click the Quit button to quit.", vbExclamation
    End If
End Sub





