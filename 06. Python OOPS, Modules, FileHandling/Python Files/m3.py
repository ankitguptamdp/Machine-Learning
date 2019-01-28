if __name__=="__main__":
	# What to do when this module is run directly
	print("M3 module "+__name__)
else:
	#Specify what to do when this module is imported
	print("Else Block ",__name__)