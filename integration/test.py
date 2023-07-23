class Test:
    
    _flag = True
    @property
    def flag(self):
        return self._flag
    
    @flag.setter
    def flag(self, val):
        self._flag = val
        print(self._flag)
        import sys
        sys.modules[__name__].__dict__['prop'] = self.prop
        
    @property
    def prop(self):
        if self.flag:
            return 'yes'
        else:
            return 'no'
        
t = Test()
prop = t.prop
