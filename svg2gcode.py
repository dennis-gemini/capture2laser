import re
import ctypes
import itertools
import xml.etree.cElementTree as ET
import svgpathtools

class Command:
    linefeed = "\n"

    def __init__(self, machine, canvas):
        self._machine = machine
        self._canvas  = canvas


    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self) + ")"


    def normalized_location(self, location=None, default=None, x=None, y=None, z=None):
        normalized = {}
        if x is not None:
            normalized["x"] = x
        if y is not None:
            normalized["y"] = y
        if z is not None:
            normalized["z"] = z

        if default is not None:
            if "x" in default:
                normalized["x"] = default["x"]
            if "y" in default:
                normalized["y"] = default["y"]
            if "z" in default:
                normalized["z"] = default["z"]

        if location is not None:
            if type(location) is dict:
                if "x" in location:
                    normalized["x"] = location["x"]
                if "y" in location:
                    normalized["y"] = location["y"]
                if "z" in location:
                    normalized["z"] = location["z"]
            elif type(location) in (tuple, list):
                if len(location) > 0:
                    normalized["x"] = location[0]
                if len(location) > 1:
                    normalized["y"] = location[1]
                if len(location) > 2:
                    normalized["z"] = location[2]
            elif type(location) is complex:
                normalized["x"] = location.real
                normalized["y"] = location.imag

        return normalized


    def join_commands(self, *cmds):
        expanded = []

        for cmd in cmds:
            if type(cmd) in (tuple, list):
                expanded.append(self.join_commands(*cmd))
            elif cmd:
                expanded.append(str(cmd))

        return self.linefeed.join(expanded)


    def setup(self):
        return self.join_commands(self._machine.setup())


    def begin(self, offset_x=None, offset_y=None, offset_z=None):
        return self.join_commands(self._machine.begin(offset_x, offset_y, offset_z))


    def end(self):
        return self.join_commands(self._machine.end())


    def moveTo(self, x=None, y=None, z=None, f=None):
        return self.join_commands(self._machine.moveTo(x, y, z, f, canvas=self._canvas))


    def lineTo(self, x=None, y=None, z=None, f=None):
        return self.join_commands(self._machine.lineTo(x, y, z, f, canvas=self._canvas))


    def arcCwTo(self, x=None, y=None, i=None, j=None, f=None):
        return self.join_commands(self._machine.arcTo(x, y, i, j, f, clockwise=True, canvas=self._canvas))


    def arcCcwTo(self, x=None, y=None, i=None, j=None, f=None):
        return self.join_commands(self._machine.arcTo(x, y, i, j, f, clockwise=False, canvas=self._canvas))


class Setup(Command):
    def __init__(self, machine, canvas):
        Command.__init__(self, machine, canvas)


    def __str__(self):
        return self.setup()


class Begin(Command):
    def __init__(self, machine, canvas, offset_x=None, offset_y=None, offset_z=None):
        Command.__init__(self, machine, canvas)
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._offset_z = offset_z


    def __str__(self):
        return self.begin(self._offset_x, self._offset_y, self._offset_z)


class End(Command):
    def __init__(self, machine, canvas):
        Command.__init__(self, machine, canvas)


    def __str__(self):
        return self.end()


class MoveTo(Command):
    def __init__(self, machine, canvas, location=None, x=None, y=None, z=None):
        Command.__init__(self, machine, canvas)
        self._location = self.normalized_location(location=location, x=x, y=y, z=z)


    def __str__(self):
        return self.moveTo(**self._location)


class LineTo(Command):
    def __init__(self, machine, canvas, start=None, end=None, x0=None, y0=None, x1=None, y1=None, z=None):
        Command.__init__(self, machine, canvas)
        self._start = self.normalized_location(location=start, x=x0, y=y0, z=z)
        self._end   = self.normalized_location(location=end  , x=x1, y=y1)


    def __str__(self):
        if self._start:
            return self.join_commands(
                self.moveTo(**self._start),
                self.lineTo(**self._end)
            )
        return self.lineTo(**self._end)


class ArcTo(Command):
    def __init__(self, machine, canvas, start=None, end=None, center=None, x0=None, y0=None, x1=None, y1=None, z=None, i=None, j=None, clockwise=True):
        Command.__init__(self, machine, canvas)
        self._start     = self.normalized_location(location=start , x=x0, y=y0, z=z)
        self._end       = self.normalized_location(location=end   , x=x1, y=y1)
        self._center    = self.normalized_location(location=center, x=i , y=j)
        self._clockwise = clockwise


    def __str__(self):
        arc_args = self._end.copy()
        arc_args.update(self._center)

        if self._start:
            return self.join_commands(
                self.moveTo(**self._start),
                self.arcCwTo(**arc_args) if clockwise else self.arcCcwTo(**arc_args)
            )
        return self.arcCwTo(**arc_args) if clockwise else self.arcCcwTo(**arc_args)


class PolylineTo(Command):
    def __init__(self, machine, canvas, start_point=None, end_points=None, x=None, y=None, z=None):
        Command.__init__(self, machine, canvas)
        self._start_point = self.normalized_location(location=start_point, x=x, y=y, z=z)
        self._end_points  = []

        if end_points is not None:
            for i in range(len(end_points)):
                p = self.normalized_location(location=end_points[i])
                if p:
                    self._end_points.append(p)


    def __str__(self):
        cmd = []
        if self._start_point:
            cmd.append(self.moveTo(**self._start_point))

        for i in range(len(self._end_points)):
            cmd.append(self.lineTo(**self._end_points[i]))

        return self.join_commands(*cmd)


class Machine:
    dynamic_power_mode = False

    def __init__(self, x_max, y_max, z_max, feed_speed, travel_speed, spindle_speed, offset_x=None, offset_y=None, offset_z=None, scaled_to_fit=False, relative_mode=True):
        self._x_max         = x_max
        self._y_max         = y_max
        self._z_max         = z_max
        self._feed_speed    = feed_speed
        self._travel_speed  = travel_speed
        self._spindle_speed = spindle_speed
        self._offset_x      = abs(offset_x) if offset_x is not None else 0
        self._offset_y      = abs(offset_y) if offset_y is not None else 0
        self._offset_z      = abs(offset_z) if offset_z is not None else 0
        self._scaled_to_fit = scaled_to_fit
        self._relative_mode = relative_mode
        self._x_cur         = 0
        self._y_cur         = 0
        self._z_cur         = 0
        self._i_cur         = 0
        self._j_cur         = 0
        self._s_cur         = 0
        self._f_cur         = 0
        self._aspect        = None


    def _round_value(self, value):
        return round(float(value), 3)


    def _init_aspects(self, canvas):
        if self._aspect is None:
            if self._scaled_to_fit or self._x_max < canvas.width:
                aspect_x = self._x_max / canvas.width
            else:
                aspect_x = 1.0

            if self._scaled_to_fit or self._y_max < canvas.height:
                aspect_y = self._y_max / canvas.height
            else:
                aspect_y = 1.0

            self._aspect = min(aspect_x, aspect_y)


    def _scale_x(self, x, canvas=None):
        if canvas is None:
            return x
        self._init_aspects(canvas)
        return x * self._aspect


    def _scale_y(self, y, canvas=None):
        if canvas is None:
            return y
        self._init_aspects(canvas)
        return y * self._aspect


    def _translate_x(self, x):
        return self._round_value(x - self._x_cur if self._relative_mode else -self._x_max + x + 1)


    def _translate_y(self, y):
        return self._round_value(y - self._y_cur if self._relative_mode else -self._y_max + y + 1)


    def _translate_z(self, z):
        return -self._round_value(z - self._z_cur if self._relative_mode else z)


    def setup(self):
        return [
            #$0=10
            #$1=25
            #$2=0
            "$3=0 (Direction inversion, mask)",
            #$4=0
            #$5=0
            #$6=0
            #$10=1
            #$11=0.010
            #$12=0.002
            #$13=0
            "$22=1 (Home cycle enable, boolean)",
            "$20=1 (Soft limits enable, boolean)",
            "$21=1 (Hard limits enable, boolean)",
            "$23=3 (Homing direction invert, mask)",
            #$24=25.000
            #$25=500.000
            #$26=250
            "$27=2.000",
            #$30=1000
            #$31=0
            "$32=1 (Laser mode, boolean)",
            #$100=1600.000
            #$101=1600.000
            #$102=1600.000
            "$110=1500.000",
            "$111=1500.000",
            #$112=800.000
            "$120=45.000",
            "$121=45.000",
            #$122=30.000
	    "$130={} (X-axis maximum travel, millimeters)".format(self._x_max),
	    "$131={} (Y-axis maximum travel, millimeters)".format(self._y_max),
	    "$132={} (Z-axis maximum travel, millimeters)".format(self._z_max),
        ]


    def begin(self, offset_x=None, offset_y=None, offset_z=None):
        result = [
            "$X",
            "$H",
            "G21",
        ]

        if self._relative_mode:
            result.append("G91")
        else:
            result.append("G90")

        if offset_x is not None:
            self._offset_x = abs(self._round_value(offset_x))
        if offset_y is not None:
            self._offset_y = abs(self._round_value(offset_y))
        if offset_z is not None:
            self._offset_z = abs(self._round_value(offset_z))

        init_move_cmd = ["G0"]
        init_move_cmd.append("X{}".format(self._translate_x(self._offset_x)))
        init_move_cmd.append("Y{}".format(self._translate_y(self._offset_y)))
        init_move_cmd.append("Z{}".format(self._translate_z(self._offset_z)))
        result.append(" ".join(init_move_cmd))

        result.extend(self.spindleInit())
        return result


    def end(self):
        result = []

        result.extend(self.moveTo(x=self._offset_x, y=self._offset_y))
        result.extend(self.spindleFinish())
        result.extend([
            "G21",
            "G90",
            "G0 X0 Y0",
        ])
        return result


    def spindleInit(self):
        self._s_cur == 0

        if self.dynamic_power_mode:
            return ["M4 S0"]
        # constant power
        return ["M3 S0"]


    def spindleFinish(self):
        if self._s_cur == 0:
            return []

        self._s_cur = 0
        return ["M5"]


    def spindleOff(self):
        if self._s_cur == 0:
            return False
        self._s_cur = 0
        return True


    def spindleOn(self, speed=None):
        if speed is None or speed == self._spindle_speed:
            if self._s_cur == self._spindle_speed:
                return False
        else:
            self._spindle_speed = speed

        self._s_cur = self._spindle_speed 
        return True


    def moveTo(self, x=None, y=None, z=None, f=None, canvas=None):
        result = []

        x  = self._x_cur        if x is None else self._round_value(self._scale_x(x, canvas))
        y  = self._y_cur        if y is None else self._round_value(self._scale_y(y, canvas))
        z  = self._z_cur        if z is None else self._round_value(z)
        f  = self._travel_speed if f is None else self._round_value(f)
        dx = self._round_value(x - self._x_cur)
        dy = self._round_value(y - self._y_cur)
        dz = self._round_value(z - self._z_cur)

        if any([dx, dy, dz]):
            cmd = ["G0"]

            if dx:
                cmd.append("X{}".format(self._translate_x(x)))
                self._x_cur = x
            if dy:
                cmd.append("Y{}".format(self._translate_y(y)))
                self._y_cur = y
            if dz:
                cmd.append("Z{}".format(self._translate_z(z)))
                self._z_cur = z

            if self._travel_speed != f:
                self._travel_speed = f

            if self._f_cur != self._travel_speed:
                self._f_cur = self._travel_speed
                cmd.append("F{}".format(self._travel_speed))

            if self.spindleOff():
                cmd.append("S0")

            result.append(" ".join(cmd))

        return result


    def lineTo(self, x=None, y=None, z=None, f=None, canvas=None):
        result = []

        x  = self._x_cur      if x is None else self._round_value(self._scale_x(x, canvas))
        y  = self._y_cur      if y is None else self._round_value(self._scale_y(y, canvas))
        z  = self._z_cur      if z is None else self._round_value(z)
        f  = self._feed_speed if f is None else self._round_value(f)
        dx = self._round_value(x - self._x_cur)
        dy = self._round_value(y - self._y_cur)
        dz = self._round_value(z - self._z_cur)

        if any([dx, dy, dz]):
            cmd = ["G1"]

            if dx:
                cmd.append("X{}".format(self._translate_x(x)))
                self._x_cur = x
            if dy:
                cmd.append("Y{}".format(self._translate_y(y)))
                self._y_cur = y
            if dz:
                cmd.append("Z{}".format(self._translate_z(z)))
                self._z_cur = z

            if self._feed_speed != f:
                self._feed_speed = f

            if self._f_cur != self._feed_speed:
                self._f_cur = self._feed_speed
                cmd.append("F{}".format(self._feed_speed))

            if self.spindleOn():
                cmd.append("S{}".format(self._s_cur))

            result.append(" ".join(cmd))

        return result

    def arcTo(self, x=None, y=None, i=None, j=None, f=None, clockwise=True, canvas=None):
        result = []

        x  = self._x_cur      if x is None else self._round_value(self._scale_x(x, canvas))
        y  = self._y_cur      if y is None else self._round_value(self._scale_y(y, canvas))
        i  = self._i_cur      if i is None else self._round_value(self._scale_x(i, canvas))
        j  = self._j_cur      if j is None else self._round_value(self._scale_y(j, canvas))
        f  = self._feed_speed if f is None else self._round_value(f)

        dx = self._round_value(x - self._x_cur)
        dy = self._round_value(y - self._y_cur)
        di = self._round_value(i - self._i_cur)
        dj = self._round_value(j - self._j_cur)

        if any([dx, dy, di, dj]):
            cmd = []

            if clockwise:
                cmd.append("G2")
            else:
                cmd.append("G3")

            if di:
                cmd.append("I{}".format(self._translate_x(i)))
                self._i_cur = i
            if dj:
                cmd.append("J{}".format(self._translate_y(j)))
                self._j_cur = j
            if dx:
                cmd.append("X{}".format(self._translate_x(x)))
                self._x_cur = x
            if dy:
                cmd.append("Y{}".format(self._translate_y(y)))
                self._y_cur = y

            if self._feed_speed != f:
                self._feed_speed = f

            if self._f_cur != self._feed_speed:
                self._f_cur = self._feed_speed
                cmd.append("F{}".format(self._feed_speed))

            if self.spindleOn():
                cmd.append("S{}".format(self._s_cur))

            result.append(" ".join(cmd))

        return result


    def position(self):
        return (self._x_cur, self._y_cur, self._z_cur)


    def center(self):
        return (self._i_cur, self._j_cur)


    def diff(self, abs_x, abs_y, abs_z):
        return (abs_x - self._x_cur, abs_y - self._y_cur, abs_z - self._z_cur)



class SVG:
    enable_transform = True
    enable_translate = False
    enable_scale     = True
    steps_for_curve  = 12

    def __init__(self, filepath):
        self._filepath = filepath
        self._paths, self._attrs, self._info = svgpathtools.svg2paths2(self._filepath)

        w, x_unit, x_scale = self._parse_by_unit(self._info["width" ])
        h, y_unit, y_scale = self._parse_by_unit(self._info["height"])

        self._canvas = {
            "width"      : w,
            "height"     : h,
            "x_unit"     : x_unit,
            "y_unit"     : y_unit,
            "x_scale"    : x_scale,
            "y_scale"    : y_scale,
            "zoom_x"     : 0,
            "zoom_y"     : 0,
            "zoom_width" : w,
            "zoom_height": h,
        }

        if "viewBox" in self._info:
            try:
                (x_min, y_min, box_width, box_height) = re.split(r"\s*,\s*|\s+", self._info["viewBox"].strip())
                self._canvas["zoom_x"     ]  = float(x_min)
                self._canvas["zoom_y"     ]  = float(y_min)
                self._canvas["zoom_width" ]  = float(box_width)
                self._canvas["zoom_height"]  = float(box_height)
                self._canvas["x_scale"    ] *= self._canvas["width" ] / self._canvas["zoom_width" ]
                self._canvas["y_scale"    ] *= self._canvas["height"] / self._canvas["zoom_height"]
            except:
                pass

        if self.enable_transform:
            i = 0
            root = ET.parse(self._filepath).getroot()
            for g in root.findall("svg:g", {'svg':'http://www.w3.org/2000/svg'}):
                self._attrs[i]["transform"] = g.attrib.get("transform", "").strip()
                i += 1

            self._transform_sequence = []
            for i in range(len(self._attrs)):
                self._transform_sequence.append(self._convert_transform(self._attrs[i]["transform"]))
        else:
            self._transform_sequence = [[]] * len(self._attrs)


    def _parse_by_unit(self, sval):
        value, unit, scale = None, None, None

        try:
            matched = re.match('^([0-9.]+)([a-zA-Z]*)$', sval.strip())
            if matched:
                value = float(matched.group(1))
                unit  = matched.group(2)

                if unit in ("", "px"):
                    scale = 25.4 / 96   # mm/px ( 25.4mm/inch / 96dpi )
                elif unit == "pt":
                    scale = 25.4 / 72   # mm/pt ( 25.4mm/inch / 72dpi )
                elif unit == "pc":
                    scale = 25.4 / 6    # mm/pc ( 25.4mm/inch / 6dpi  )
                elif unit == "cm":
                    scale = 10.0
                elif unit == "mm":
                    scale = 1.0
                elif unit == "in":
                    scale = 25.4
                else:
                    value, unit = None, None
        except:
            pass

        return (value, unit, scale)


    def _normalized_x_in_mm(self, x_in_unit):
        return (x_in_unit - self._canvas["zoom_x"]) * self._canvas["x_scale"]


    def _normalized_y_in_mm(self, y_in_unit):
        return (y_in_unit - self._canvas["zoom_y"]) * self._canvas["y_scale"]


    def _normalized_coord_in_mm(self, a, b=None):
        if type(a) == complex:
            x, y = a.real, a.imag
        elif type(a) in (tuple, list):
            x, y = a[:2]
        else:
            x, y = a, b

        return (self._normalized_x_in_mm(x), self._normalized_y_in_mm(y))


    def _lambda_translate(self, args):
        if not self.enable_translate:
            return None

        tx, ty = args

        if tx[-1] == "%":
            tx = float(tx[:-1])
            def _translate_x(x):
                return x + self._canvas["width"] * (tx / 100.0)
        else:
            tx = float(tx)
            def _translate_x(x):
                return x + tx

        if ty[-1] == "%":
            ty = float(ty[:-1])
            def _translate_y(y):
                return y + self._canvas["height"] * (ty / 100.0)
        else:
            ty = float(ty)
            def _translate_y(y):
                return y + ty

        def _translate_coord(a, b=None):
            if type(a) is complex:
                x, y = a.real, a.imag
            elif type(a) in (tuple, list):
                x, y = a[:2]
            else:
                x, y = a, b

            return (_translate_x(x), _translate_y(y))

        return (_translate_coord, _translate_x, _translate_y)


    def _lambda_scale(self, args):
        if not self.enable_scale:
            return None

        sx, sy = args

        sx = float(sx)
        sy = float(sy)
        #ox = 0 if sx >= 0 else self._canvas["width" ]
        #oy = 0 if sy >= 0 else self._canvas["height"]
        sx = abs(sx)
        sy = abs(sy)

        def _scale_x(x):
            return x * sx #+ ox

        def _scale_y(y):
            return y * sy #+ oy

        def _scale_coord(a, b=None):
            if type(a) is complex:
                x, y = a.real, a.imag
            elif type(a) in (tuple, list):
                x, y = a[:2]
            else:
                x, y = a, b

            return (_scale_x(x), _scale_y(y))

        return (_scale_coord, _scale_x, _scale_y)


    def _convert_transform(self, spec):
        STATE_NOOP   = 0
        STATE_OP     = 1
        STATE_OP_END = 2
        STATE_ARGS   = 3
        state        = STATE_NOOP
        operator     = []
        args         = []
        cur          = []
        transform    = []

        def _add_functors_to_trnasform(self):
            functors = eval("self._lambda_%s" % "".join(operator))(args)
            if functors:
                transform.append(functors)

        for c in spec:
            if ord(c) <= 0x20:
                if state == STATE_NOOP:
                    pass
                elif state == STATE_OP:
                    state == STATE_OP_END
                elif state == STATE_OP_END:
                    pass
                elif state == STATE_ARGS:
                    if cur:
                        args.append("".join(cur))
                        cur = []
                continue

            if c.isalpha():
                if state == STATE_NOOP:
                    state = STATE_OP
                    operator.append(c)
                elif state == STATE_OP:
                    operator.append(c)
                elif state == STATE_OP_END:
                    state = STATE_NOOP
                    _add_functors_to_trnasform(self)
                    operator = []
                    args     = []
                    cur      = []
                elif state == STATE_ARGS:
                    return None

                continue

            if c.isdigit() or c in "+-.%":
                if state == STATE_NOOP:
                    return None
                elif state == STATE_OP:
                    return None
                elif state == STATE_OP_END:
                    return None
                elif state == STATE_ARGS:
                    cur.append(c)

                continue

            if c == "(":
                if state == STATE_NOOP:
                    return None
                elif state == STATE_OP:
                    state = STATE_ARGS
                elif state == STATE_OP_END:
                    state = STATE_ARGS
                elif state == STATE_ARGS:
                    return None

                continue
            
            if c == ")":
                if state == STATE_NOOP:
                    return None
                elif state == STATE_OP:
                    return None
                elif state == STATE_OP_END:
                    return None
                elif state == STATE_ARGS:
                    state = STATE_NOOP

                    if cur:
                        args.append("".join(cur))

                    _add_functors_to_trnasform(self)
                    operator = []
                    args     = []
                    cur      = []

                continue

            if c == ",":
                if state == STATE_NOOP:
                    return None
                elif state == STATE_OP:
                    return None
                elif state == STATE_OP_END:
                    return None
                elif state == STATE_ARGS:
                    if cur:
                        args.append("".join(cur))
                        cur = []

                continue


        if state == STATE_NOOP:
            pass
        elif state == STATE_OP:
            if operator:
                _add_functors_to_trnasform(self)
        elif state == STATE_OP_END:
            if operator:
                _add_functors_to_trnasform(self)
        elif state == STATE_ARGS:
            return None

        return transform


    def _transform_x(self, ndx, x):
        for transform in self._transform_sequence[ndx]:
            x = transform[1](x)
        return self._normalized_x_in_mm(x)


    def _transform_y(self, ndx, y):
        for transform in self._transform_sequence[ndx]:
            y = transform[2](y)
        return self._normalized_y_in_mm(y)


    def _transform_coord(self, ndx, a, b=None):
        if type(a) is complex:
            x, y = a.real, a.imag
        elif type(a) in (tuple, list):
            x, y = a[:2]
        else:
            x, y = a, b

        for transform in self._transform_sequence[ndx]:
            x, y = transform[0](x, y)

        return self._normalized_coord_in_mm(x, y)


    @property
    def width(self):
        return self._normalized_x_in_mm(self._canvas["width"])


    @property
    def height(self):
        return self._normalized_y_in_mm(self._canvas["height"])


    def generate_gcode(self, machine):
        gcode = []
        gcode.append(Begin(machine, self))

        for i, path in enumerate(self._paths):
            for j, segment in enumerate(path):
                if type(segment) is svgpathtools.path.Line:
                    x0, y0 = self._transform_coord(i, segment.start)
                    x1, y1 = self._transform_coord(i, segment.end  )
                    gcode.append(LineTo(machine, self, x0=x0, y0=y0, x1=x1, y1=y1))
                elif type(segment) is svgpathtools.path.Arc and (segment.radius.real == segment.radius.imag):
                    x0, y0 = self._transform_coord(i, segment.start )
                    x1, y1 = self._transform_coord(i, segment.end   )
                    cx, cy = self._transform_coord(i, segment.center)
                    gcode.append(ArcTo(machine, self, x0=x0, y0=y0, x1=x1, y1=y1, i=cx, j=cy, clockwise=segment.sweep))
                else:
                    x0, y0 = self._transform_coord(i, segment.start)
                    points = []

                    for k in range(self.steps_for_curve + 1):
                        points.append(self._transform_coord(i, segment.point(k / float(self.steps_for_curve))))
                    gcode.append(PolylineTo(machine, self, x=x0, y=y0, end_points=points))

        gcode.append(End(machine, self))
        return gcode

if __name__ == "__main__":
    # 1500 -> 7min
    # 1200 -> 10min
    # 1000 -> 13min
    # 900 -> 14min
    # 800 -> 9min
    # 700 -> 10min
    # 600 -> 10min
    # 500 -> 9min
    import sys
    print "\n".join([str(x) for x in SVG(sys.argv[1]).generate_gcode(Machine(285.0, 170.0, 38.0, 200.0, 1500.0, 255.0, offset_z=None, scaled_to_fit=False, relative_mode=False))])

